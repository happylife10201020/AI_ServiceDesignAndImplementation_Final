"""PDF에서 텍스트를 뽑고 gpt-4o로 다듬는 스크립트.

두 단계로 돈다. 먼저 pypdf로 원문을 뽑아 output/<이름>.raw.txt에 저장하고,
그 다음 gpt-4o로 줄바꿈/오탈자/문단만 정리해서 output/<이름>.txt에 저장한다.
gpt-4o 호출이 실패하면 원문을 그대로 저장한다. 키나 .env 내용은 로그에 남기지 않는다.

extract_raw는 rag.py의 인덱싱에서도 재사용한다.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from tqdm import tqdm

try:
    from openai import (
        APIConnectionError,
        APIError,
        APITimeoutError,
        OpenAI,
        RateLimitError,
    )
except ImportError:  # openai 패키지가 없을 때 방어
    OpenAI = None  # type: ignore[assignment]
    APIError = APITimeoutError = RateLimitError = APIConnectionError = Exception  # type: ignore[assignment]


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

# gpt-4o 호출 한 번에 넣을 최대 글자 수. 넘으면 문단 단위로 쪼갠다.
MAX_CHARS_PER_CALL = 12000

SYSTEM_PROMPT = (
    "당신은 텍스트 정리 전문가입니다. PDF에서 추출한 원문을 받아 다음만 합니다.\n"
    "1. 깨진 줄바꿈을 자연스러운 문단으로 복원\n"
    "2. 명백한 오탈자(글자 깨짐, 중복 공백 등) 교정\n"
    "3. 문단 구조 정리\n\n"
    "요약, 축약, 생략, 새 문장 창작, 원문에 없는 해석 추가는 하지 마세요. "
    "내용을 그대로 보존한 정리된 텍스트만 출력하고, 안내 문구나 코드블록 표시는 붙이지 마세요."
)


@dataclass
class Config:
    api_key: str
    model: str


def load_config() -> Config:
    # .env를 읽고 OPENAI_API_KEY가 없으면 종료. 키 값은 절대 출력하지 않는다.
    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")

    if not api_key:
        logging.error("OPENAI_API_KEY가 없습니다. .env를 확인하세요.")
        sys.exit(1)

    return Config(api_key=api_key, model=model)


def find_pdfs() -> list[Path]:
    return sorted(BASE_DIR.glob("*.pdf"))


def extract_raw(pdf_path: Path) -> str | None:
    """PDF 전체 페이지에서 텍스트를 뽑는다. 손상/암호화로 못 읽으면 None."""
    try:
        reader = PdfReader(pdf_path)

        if reader.is_encrypted:
            # 빈 비밀번호로 열어보고 안 되면 건너뛴다.
            try:
                if reader.decrypt("") == 0:
                    logging.warning("암호화 PDF라 건너뜀: %s", pdf_path.name)
                    return None
            except Exception:
                logging.warning("암호화 PDF라 건너뜀: %s", pdf_path.name)
                return None

        pages_text = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception as exc:  # 페이지 단위 손상 방어
                logging.warning("%s: 페이지 추출 오류, 빈 문자열 사용 (%s)", pdf_path.name, exc)
                pages_text.append("")

        return "\n\n".join(pages_text)

    except (PdfReadError, OSError, ValueError) as exc:
        logging.warning("손상 PDF라 건너뜀: %s (%s)", pdf_path.name, exc)
        return None
    except Exception as exc:  # 예상 못 한 pypdf 오류 방어
        logging.warning("알 수 없는 오류로 건너뜀: %s (%s)", pdf_path.name, exc)
        return None


def _split_into_chunks(text: str, max_chars: int) -> list[str]:
    # 문단(\n\n) 경계로 max_chars 이하 청크로 나눈다. 문단 하나가 더 크면 그대로 둔다.
    if len(text) <= max_chars:
        return [text] if text else []

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # 재결합 시 붙는 "\n\n" 보정

        if current and current_len + para_len > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def refine_with_gpt4o(client: "OpenAI", model: str, raw_text: str) -> str | None:
    """gpt-4o로 원문의 줄바꿈/오탈자/문단만 정리한다. 실패하면 None(호출부에서 원문 저장)."""
    if not raw_text.strip():
        return raw_text

    chunks = _split_into_chunks(raw_text, MAX_CHARS_PER_CALL)
    refined_chunks: list[str] = []

    for chunk in chunks:
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": chunk},
                ],
            )
        except (APIError, APITimeoutError, RateLimitError, APIConnectionError) as exc:
            logging.warning("OpenAI 호출 실패, 원문 사용 (%s)", type(exc).__name__)
            return None
        except Exception as exc:  # 예상 못 한 SDK 오류 방어
            logging.warning("OpenAI 호출 중 알 수 없는 오류, 원문 사용 (%s)", exc)
            return None

        content = None
        if response.choices:
            content = response.choices[0].message.content

        if not content or not content.strip():
            logging.warning("gpt-4o 빈 응답, 원문 사용")
            return None

        refined_chunks.append(content.strip())

    return "\n\n".join(refined_chunks)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def process_one(pdf_path: Path, client: "OpenAI", model: str) -> str:
    """PDF 하나를 추출+정리한다. 결과는 success / fallback / failed."""
    stem = pdf_path.stem

    raw_text = extract_raw(pdf_path)
    if raw_text is None:
        return "failed"

    raw_path = OUTPUT_DIR / f"{stem}.raw.txt"
    save_text(raw_path, raw_text)
    logging.info("추출 완료: %s", raw_path.name)

    refined_text = refine_with_gpt4o(client, model, raw_text)

    final_path = OUTPUT_DIR / f"{stem}.txt"
    if refined_text is not None:
        save_text(final_path, refined_text)
        logging.info("정리 완료: %s", final_path.name)
        return "success"

    save_text(final_path, raw_text)
    logging.warning("정리 실패, 원문 저장: %s", final_path.name)
    return "fallback"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = load_config()

    if OpenAI is None:
        logging.error("openai 패키지를 불러올 수 없습니다. requirements.txt를 확인하세요.")
        return 1

    client = OpenAI(api_key=config.api_key)

    pdf_paths = find_pdfs()
    if not pdf_paths:
        logging.info("처리할 PDF가 없습니다: %s", BASE_DIR)
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("PDF %d개 발견", len(pdf_paths))

    success_count = 0
    fallback_count = 0
    failed_count = 0

    for pdf_path in tqdm(pdf_paths, desc="PDF 처리", unit="file"):
        logging.info("처리 시작: %s", pdf_path.name)
        try:
            status = process_one(pdf_path, client, config.model)
        except Exception as exc:  # 파일 하나 오류가 전체를 멈추지 않게
            logging.error("알 수 없는 오류로 건너뜀: %s (%s)", pdf_path.name, exc)
            status = "failed"

        if status == "success":
            success_count += 1
        elif status == "fallback":
            fallback_count += 1
        else:
            failed_count += 1

    logging.info(
        "요약 - 전체 %d, 정리 성공 %d, 원문 저장 %d, 실패 %d",
        len(pdf_paths),
        success_count,
        fallback_count,
        failed_count,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
