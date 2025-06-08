import os
import json
import time
import argparse 

# Import necessary libraries for Kaggle Secrets and the new client
try:
    from kaggle_secrets import UserSecretsClient
    KAGGLE_SECRETS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    KAGGLE_SECRETS_AVAILABLE = False
    print("⚠️  Warning: Kaggle 'UserSecretsClient' not found. Will use hardcoded API keys if available.")

from google import genai
from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
    FinishReason
)

# --- AI Prompt Definition ---
# This is a large constant, so it's fine to keep it at the global level.
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
# Sẽ được khởi tạo trong hàm main
api_keys = []
current_key_index = 0
current_key_uses = 0

# --- Hàm xử lý API ---
def enrich_text_with_gemini(text_to_enrich, args):
    """
    Enriches text by calling the Gemini API and handles API key rotation.
    """
    global current_key_index, current_key_uses, api_keys

    if not api_keys:
        print("❌ Error: API keys list is empty. Cannot make a request.")
        return text_to_enrich

    # Chuẩn bị client với key hiện tại
    client = genai.Client(api_key=api_keys[current_key_index])
    full_input = f"{AI_PROMPT}\n\n{text_to_enrich}"

    try:
        response = client.models.generate_content(
            model=args.model,
            contents=[full_input]
        )

        if not response.candidates:
            print(f"API response empty or blocked for text: '{text_to_enrich[:80]}...'")
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"Blocked reason: {response.prompt_feedback.block_reason}")
            return text_to_enrich

        candidate = response.candidates[0]
        if candidate.finish_reason != FinishReason.STOP:
            print(f"API finished with non-STOP reason: {candidate.finish_reason.name} for text: '{text_to_enrich[:80]}...'")
            return text_to_enrich

        if not hasattr(candidate, 'content') or not hasattr(candidate.content, 'parts') or not candidate.content.parts:
            print(f"API finished with STOP but no content parts found for text: '{text_to_enrich[:80]}...'")
            return text_to_enrich

        enriched_text = candidate.content.parts[0].text.strip()
        print(f"SUCCESSFUL! {enriched_text}\n")

        # Tăng counter sau mỗi lần gọi thành công
        current_key_uses += 1
        
        # Kiểm tra nếu đã dùng đủ số lần cho phép
        if current_key_uses >= args.requests_per_key:
            current_key_index = (current_key_index + 1) % len(api_keys)
            current_key_uses = 0
            print(f"🔄 Switching to API key index {current_key_index}")

        return enriched_text
    except Exception as e:
        print(f"API call failed for text: '{text_to_enrich[:80]}...' Error: {e}")
        time.sleep(5)
        return text_to_enrich

# --- Main Processing Logic ---
def main(args):
    """
    Main function to load, process, and save the data.
    """
    global api_keys # Khai báo để có thể thay đổi biến toàn cục

    # --- API Key Setup ---
    # Ưu tiên lấy từ Kaggle Secrets, nếu không có thì dùng key hardcode
    if KAGGLE_SECRETS_AVAILABLE:
        try:
            user_secrets = UserSecretsClient()
            # Giả sử secret chứa một chuỗi JSON của các key
            # Ví dụ: ["key1", "key2", "key3"]
            secret_string = user_secrets.get_secret(args.secret_name)
            api_keys = json.loads(secret_string)
            print(f"✅ Successfully loaded {len(api_keys)} API keys from Kaggle secret '{args.secret_name}'.")
        except Exception as e:
            print(f"⚠️ Could not load keys from Kaggle secret '{args.secret_name}'. Error: {e}")
            print("Trying hardcoded fallback keys...")
    
    if not api_keys:
        # Danh sách key dự phòng nếu Kaggle Secrets không hoạt động hoặc không có sẵn
        api_keys = [
            'YOUR_API_KEY_1_HERE', # <-- THAY KEY CỦA BẠN VÀO ĐÂY
            'YOUR_API_KEY_2_HERE', # <-- THAY KEY CỦA BẠN VÀO ĐÂY
        ]
        # Xóa các key mẫu không hợp lệ
        api_keys = [key for key in api_keys if 'YOUR_API_KEY' not in key]
        
        if api_keys:
            print(f"✅ Using {len(api_keys)} hardcoded fallback API keys.")
        else:
            print("❌ Critical Error: No API keys found in Kaggle Secrets or hardcoded list. Exiting.")
            return

    # --- Data Processing ---
    print(f"Loading data from {args.input}")
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} encounters.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.input}. Please ensure it is valid JSON.")
        return
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return

    processed_count = 0
    for entry in data:
        if "query_content_en" in entry and isinstance(entry["query_content_en"], str) and entry["query_content_en"]:
            original_text = entry["query_content_en"]
            encounter_id = entry.get("encounter_id", "N/A")
            print(f"Processing {encounter_id} ({processed_count + 1}/{len(data)})")

            enriched_text = enrich_text_with_gemini(original_text, args)
            entry["query_content_en"] = enriched_text
            processed_count += 1

            if processed_count % args.checkpoint_freq == 0:
                print(f"Saving checkpoint to {args.output}...")
                try:
                    with open(args.output, 'w', encoding='utf-8') as outfile:
                        json.dump(data, outfile, indent=4, ensure_ascii=False)
                    print("Checkpoint saved.")
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")
        else:
            print(f"Warning: 'query_content_en' not found, empty or not a string in entry {entry.get('encounter_id', 'N/A')}. Skipping.")

    print(f"Finished processing {processed_count} entries.")
    print(f"Saving final enriched data to {args.output}")
    try:
        with open(args.output, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False)
        print("Processing complete. Enriched data saved.")
    except Exception as e:
        print(f"Error saving the final output file: {e}")

# --- Argument Parser Setup ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich medical text in a JSON file using the Gemini API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Hiển thị giá trị mặc định trong help
    )

    parser.add_argument(
        "-i", "--input",
        default="/kaggle/input/imageclef-2025-vqa/valid_ht_v2.json",
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "-o", "--output",
        default="val_enriched_output.json",
        help="Path to the output JSON file."
    )
    parser.add_argument(
        "-m", "--model",
        default="gemini-2.5-flash-preview-04-17", # Sử dụng "latest" để linh hoạt hơn
        help="The ID of the Gemini model to use for enrichment."
    )
    parser.add_argument(
        "--secret-name",
        default="gemini-api-keys", # Tên secret chứa danh sách key
        help="The name of the Kaggle secret containing a JSON list of API keys."
    )
    parser.add_argument(
        "--requests-per-key",
        type=int,
        default=15,
        help="Number of successful API calls before switching to the next key."
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10,
        help="How often (number of entries) to save a checkpoint of the output file."
    )
    
    # Phân tích các đối số từ dòng lệnh
    args = parser.parse_args()
    
    # Gọi hàm chính với các đối số đã được phân tích
    main(args)
