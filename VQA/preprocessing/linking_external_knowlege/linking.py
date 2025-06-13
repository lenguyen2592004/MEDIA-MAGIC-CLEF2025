import os
import json
import time
import argparse 
import sys

# Import necessary libraries for Kaggle Secrets and the new client
try:
    from kaggle_secrets import UserSecretsClient
    KAGGLE_SECRETS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    KAGGLE_SECRETS_AVAILABLE = False
    # Không in cảnh báo ở đây nữa, sẽ xử lý trong logic tải key

from google import genai
from google.genai.types import (
    FinishReason
)

# --- AI Prompt Definition ---
# Hằng số này lớn, giữ ở phạm vi toàn cục là hợp lý.
AI_PROMPT = """
ROLE: AI Medical Concept Enrichment Specialist
CONTEXT: You are tasked with processing text content, typically image captions or descriptions (query_content_en field) related to medical observations, often within a Visual Question Answering (VQA) context for medicine, especially but not limited to Dermatology. Your goal is to enhance this text by identifying specific medical terms and appending concise, accurate definitions. This prompt is designed for the Qwen2 VL Instruct model.
OBJECTIVE: To enrich the input text by identifying relevant medical terms (including but not limited to Dermatology) and appending their brief, accurate definitions immediately after the term, formatted as Term [Definition]. Definitions for Dermatology-specific terms should prioritize consistency with DermNet NZ (site:dermnetnz.org). Definitions for other medical terms (diseases, symptoms, findings, anatomical locations, procedures, relevant medications) should be consistent with standard medical knowledge and ontologies like SNOMED CT(site:https://www.snomed.org/) or UMLS(site:https://www.nlm.nih.gov/research/umls/index.html) or wikidoc(site:https://www.wikidoc.org/index.php/Main_Page), use your crawl skills to get information.
INPUT:
A single string of text representing the value of a query_content_en field or similar medical text description.
OUTPUT:
The modified string of text, with relevant medical terms enriched as specified. The overall structure and non-relevant parts of the original text must remain unchanged.
CONSTRAINTS:
Scope: Enrich specific medical terms. This includes:
- Names of diseases or conditions (e.g., psoriasis, diabetes mellitus).
- Specific symptoms or clinical findings (e.g., macule, papule, edema, fever, erythema).
- Relevant anatomical locations (e.g., epidermis, dermis, femur).
- Medical or surgical procedures (e.g., biopsy, Mohs surgery).
- Commonly referenced medications within a relevant medical context (e.g., methotrexate, metformin).
Exclusion: Do NOT enrich:
- Highly general medical terms used non-specifically (e.g., 'disease', 'patient', 'doctor', 'syndrome' unless part of a specific named syndrome).
- Very common words or non-medical terms (e.g., 'tired', 'predict', 'image', 'shows', 'left', 'right' unless part of a specific anatomical term like 'left ventricle').
- Terms that are already adequately explained by the immediate context.
Source Prioritization:
- For Dermatology terms: Prioritize definitions consistent with the style and content of DermNet NZ (site:dermnetnz.org).
- For other medical terms: Provide definitions consistent with established medical knowledge, reflecting concepts found in standard terminologies like SNOMED CT or UMLS, Wikidoc or reliable general medical sources (e.g., Mayo Clinic, Merck Manuals). Use your trained knowledge based on these types of sources.
Definition Format: Definitions must be concise, clear, and enclosed in square brackets [ ] immediately following the identified term, with a single space before the opening bracket. Term [Definition]
Definition Content: Definitions should be brief explanations (typically 1-2 short sentences) understandable in context, not exhaustive medical treatises. Focus on the core meaning relevant to the observation.
Accuracy: Ensure the definitions are medically accurate and appropriate for the context.
Multiple Terms: If multiple relevant terms exist in the input text, enrich each one independently.
Case Sensitivity: Identify terms regardless of their capitalization, but use the original capitalization of the term in the output before the bracketed definition.
No Modification Otherwise: Do not alter any other part of the input text. Maintain original punctuation and sentence structure where possible around the enrichment.
INSTRUCTIONS:
Receive the input text string.
Scan the text to identify potential specific medical keywords or phrases based on CONSTRAINT 1.
For each identified term:
a. Verify it meets the criteria for enrichment (specific medical term) and is not excluded by CONSTRAINT 2.
b. Determine if the term is primarily dermatological or general medical.
c. Generate a concise, accurate definition according to the Source Prioritization (CONSTRAINT 3): Use site:dermnetnz.org resource for Dermatology; use site:wikidoc.org concepts or general medical knowledge for others.
d. Format the definition as specified in CONSTRAINT 4 & 5.
e. Append the formatted definition [Definition] immediately after the term in the text.
If no relevant terms are found, return the original input text unmodified.
Return the fully processed text string as the output.
EXAMPLES:
Input Text (Dermatology Focus): The patient presented with severe psoriasis and was prescribed methotrexate. Also noted was a suspicious nevus on the left arm.
Output Text: The patient presented with severe psoriasis [a common, chronic inflammatory skin disease characterized by red, itchy, scaly patches] and was prescribed methotrexate [an immunosuppressant drug used for various conditions including severe psoriasis and certain cancers]. Also noted was a suspicious nevus [a mole, which is a common pigmented skin lesion, sometimes monitored for changes] on the left arm.
Input Text (Dermatology Focus): Close-up shows multiple comedones, typical of acne vulgaris.
Output Text: Close-up shows multiple comedones [blocked hair follicles; blackheads are open comedones, whiteheads are closed comedones], typical of acne vulgaris [a common skin condition where pores become blocked by hair, sebum, bacteria, and dead skin cells, often causing pimples].
Input Text (General Medical): Image shows pitting edema on the lower leg, possibly related to congestive heart failure.
Output Text: Image shows pitting edema [swelling, typically in the limbs, where pressing the skin leaves a temporary indentation] on the lower leg, possibly related to congestive heart failure [a chronic condition where the heart doesn't pump blood as effectively as it should].
Input Text (Mixed): Examination revealed jaundice and hepatomegaly. Skin biopsy confirmed primary biliary cholangitis.
Output Text: Examination revealed jaundice [yellowing of the skin and whites of the eyes caused by high bilirubin levels] and hepatomegaly [enlargement of the liver]. Skin biopsy [a procedure where a small sample of skin is removed for examination] confirmed primary biliary cholangitis [a chronic liver disease where bile ducts in the liver are slowly destroyed].
Input Text (General Exclusion): Doctors predict that he has some kind of infection, he feels tired.
Output Text: Doctors predict that he has some kind of infection [invasion and multiplication of microorganisms such as bacteria, viruses, and parasites that are not normally present within the body], he feels tired. (Note: 'predict', 'kind of', 'tired' are not enriched. 'infection' is enriched as it's a core medical concept here).
Now, process the following input text based on these instructions:
"""

# --- Biến toàn cục cho việc xoay vòng API key ---
api_keys = []
current_key_index = 0
current_key_uses = 0

# --- Hàm xử lý API ---
def enrich_text_with_gemini(text_to_enrich, model_name, requests_per_key):
    """
    Làm giàu văn bản bằng cách gọi Gemini API và xử lý việc xoay vòng key.
    """
    global current_key_index, current_key_uses, api_keys

    if not api_keys:
        print("❌ Lỗi: Danh sách API key trống. Không thể thực hiện yêu cầu.", file=sys.stderr)
        return text_to_enrich

    # Chuẩn bị client với key hiện tại
    client = genai.Client(api_key=api_keys[current_key_index])
    full_input = f"{AI_PROMPT}\n\n{text_to_enrich}"

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[full_input]
        )

        if not response.candidates:
            print(f"API response trống hoặc bị chặn cho văn bản: '{text_to_enrich[:80]}...'")
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"Lý do bị chặn: {response.prompt_feedback.block_reason}")
            return text_to_enrich

        candidate = response.candidates[0]
        if candidate.finish_reason != FinishReason.STOP:
            print(f"API kết thúc với lý do không phải STOP: {candidate.finish_reason.name} cho văn bản: '{text_to_enrich[:80]}...'")
            return text_to_enrich

        if not hasattr(candidate, 'content') or not hasattr(candidate.content, 'parts') or not candidate.content.parts:
            print(f"API kết thúc với STOP nhưng không có nội dung cho văn bản: '{text_to_enrich[:80]}...'")
            return text_to_enrich

        enriched_text = candidate.content.parts[0].text.strip()
        print(f"THÀNH CÔNG! {enriched_text}\n")

        # Tăng counter sau mỗi lần gọi thành công
        current_key_uses += 1
        
        # Kiểm tra nếu đã dùng đủ số lần cho phép
        if current_key_uses >= requests_per_key:
            current_key_index = (current_key_index + 1) % len(api_keys)
            current_key_uses = 0
            print(f"🔄 Đang chuyển sang API key ở vị trí {current_key_index}")

        return enriched_text
    except Exception as e:
        print(f"Lỗi gọi API cho văn bản: '{text_to_enrich[:80]}...' Lỗi: {e}", file=sys.stderr)
        # Chuyển sang key tiếp theo nếu có lỗi (ví dụ: key hết hạn, sai định dạng)
        current_key_index = (current_key_index + 1) % len(api_keys)
        current_key_uses = 0
        print(f"🔄 Thử chuyển sang API key tiếp theo ở vị trí {current_key_index} do có lỗi.")
        time.sleep(5)
        return text_to_enrich

def _load_api_keys(args):
    """
    Tải API keys theo thứ tự ưu tiên:
    1. Đối số từ dòng lệnh (--api-keys)
    2. Biến môi trường
    3. Kaggle Secrets
    Trả về một danh sách các key.
    """
    # 1. Ưu tiên cao nhất: Lấy từ đối số dòng lệnh
    if args.api_keys:
        print(f"✅ Đã tải {len(args.api_keys)} API key(s) từ đối số dòng lệnh.")
        return args.api_keys

    # 2. Ưu tiên thứ hai: Lấy từ biến môi trường
    env_keys_str = os.getenv(args.env_var)
    if env_keys_str:
        keys = [key.strip() for key in env_keys_str.split(',') if key.strip()]
        if keys:
            print(f"✅ Đã tải {len(keys)} API key(s) từ biến môi trường '{args.env_var}'.")
            return keys

    # 3. Ưu tiên thứ ba: Lấy từ Kaggle Secrets (nếu có)
    if KAGGLE_SECRETS_AVAILABLE:
        print(f"Đang thử tải key từ Kaggle secret '{args.secret_name}'...")
        try:
            user_secrets = UserSecretsClient()
            secret_string = user_secrets.get_secret(args.secret_name)
            keys = json.loads(secret_string)
            if isinstance(keys, list) and keys:
                print(f"✅ Đã tải {len(keys)} API key(s) từ Kaggle secret.")
                return keys
            else:
                print(f"⚠️ Secret '{args.secret_name}' không chứa danh sách key hợp lệ.")
        except Exception as e:
            print(f"⚠️ Không thể tải key từ Kaggle secret '{args.secret_name}'. Lỗi: {e}")

    # Nếu không có key nào được tìm thấy
    return []

# --- Main Processing Logic ---
def main(args):
    """
    Hàm chính để tải, xử lý và lưu dữ liệu.
    """
    global api_keys
    
    # --- Thiết lập API Key ---
    api_keys = _load_api_keys(args)
    if not api_keys:
        print("❌ Lỗi nghiêm trọng: Không có API key nào được cung cấp.", file=sys.stderr)
        print("Vui lòng cung cấp key qua đối số --api-keys, biến môi trường, hoặc Kaggle Secrets.", file=sys.stderr)
        sys.exit(1) # Thoát chương trình với mã lỗi

    # --- Xử lý dữ liệu ---
    print(f"Đang tải dữ liệu từ {args.input}")
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Đã tải {len(data)} bản ghi.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file đầu vào tại {args.input}", file=sys.stderr)
        return
    except json.JSONDecodeError:
        print(f"Lỗi: Không thể giải mã JSON từ {args.input}. Hãy đảm bảo file có định dạng JSON hợp lệ.", file=sys.stderr)
        return
    except Exception as e:
        print(f"Đã xảy ra lỗi khi tải file: {e}", file=sys.stderr)
        return

    processed_count = 0
    total_entries = len(data)
    for i, entry in enumerate(data):
        if "query_content_en" in entry and isinstance(entry["query_content_en"], str) and entry["query_content_en"]:
            original_text = entry["query_content_en"]
            encounter_id = entry.get("encounter_id", "N/A")
            print(f"--- Đang xử lý {encounter_id} ({i + 1}/{total_entries}) ---")
            print(f"Văn bản gốc: {original_text}")

            enriched_text = enrich_text_with_gemini(original_text, args.model, args.requests_per_key)
            entry["query_content_en"] = enriched_text
            processed_count += 1

            if processed_count > 0 and processed_count % args.checkpoint_freq == 0:
                print(f"💾 Đang lưu checkpoint vào {args.output}...")
                try:
                    with open(args.output, 'w', encoding='utf-8') as outfile:
                        json.dump(data, outfile, indent=4, ensure_ascii=False)
                    print("Lưu checkpoint thành công.")
                except Exception as e:
                    print(f"Lỗi khi lưu checkpoint: {e}", file=sys.stderr)
        else:
            print(f"⚠️ Cảnh báo: 'query_content_en' không tồn tại, rỗng hoặc không phải là chuỗi trong bản ghi {entry.get('encounter_id', 'N/A')}. Bỏ qua.")

    print(f"\n--- HOÀN TẤT ---")
    print(f"Đã xử lý {processed_count} bản ghi.")
    print(f"Đang lưu kết quả cuối cùng vào {args.output}")
    try:
        with open(args.output, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False)
        print("✅ Xử lý hoàn tất. Dữ liệu đã được lưu.")
    except Exception as e:
        print(f"Lỗi khi lưu file đầu ra cuối cùng: {e}", file=sys.stderr)

# --- Argument Parser Setup ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Làm giàu văn bản y tế trong file JSON bằng Gemini API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Đường dẫn đến file JSON đầu vào."
    )
    parser.add_argument(
        "-o", "--output",
        default="enriched_output.json",
        help="Đường dẫn đến file JSON đầu ra."
    )
    parser.add_argument(
        "--api-keys",
        nargs='+', # Chấp nhận một hoặc nhiều giá trị
        help="Một hoặc nhiều API key của Gemini. Đây là tùy chọn có ưu tiên cao nhất."
    )
    parser.add_argument(
        "-m", "--model",
        default="gemini-1.5-flash-latest",
        help="ID của model Gemini sẽ sử dụng."
    )
    parser.add_argument(
        "--secret-name",
        default="gemini-api-keys",
        help="Tên của Kaggle secret chứa danh sách API key (định dạng JSON)."
    )
    parser.add_argument(
        "--env-var",
        default="GEMINI_API_KEYS",
        help="Tên biến môi trường chứa các API key (phân tách bằng dấu phẩy)."
    )
    parser.add_argument(
        "--requests-per-key",
        type=int,
        default=50, # Flash model thường có giới hạn cao hơn
        help="Số lượng yêu cầu API thành công trước khi chuyển sang key tiếp theo."
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=20,
        help="Tần suất (số bản ghi) lưu checkpoint của file đầu ra."
    )
    
    args = parser.parse_args()
    main(args)
