import requests
import json
import concurrent.futures

# =====================================================================
# HÀM CƠ SỞ: GỌI API VÀ LẤY DỮ LIỆU ĐA LUỒNG (MULTI-THREADING)
# =====================================================================
def fetch_single_drug_record(drug_name):
    """Hàm lõi gọi API cho 1 loại thuốc. Trả về tuple: (drug_name, record, error)"""
    url = f'https://api.fda.gov/drug/label.json?search=openfda.generic_name:"{drug_name}"+openfda.brand_name:"{drug_name}"&limit=1'
    try:
        response = requests.get(url).json()
        if 'error' in response:
            return drug_name, None, f"No FDA drug label data found for: {drug_name}"
        return drug_name, response['results'][0], None
    except Exception as e:
        return drug_name, None, f"FDA API connection error: {str(e)}"

def fetch_multiple_drug_records(drug_names: list):
    """
    Sử dụng ThreadPoolExecutor để gọi API song song cho toàn bộ danh sách thuốc.
    Trả về Dictionary: { "ten_thuoc": {"record": ..., "error": ...} }
    """
    results = {}
    # max_workers=5 nghĩa là có thể kéo tối đa 5 loại thuốc cùng 1 lúc
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Gửi toàn bộ request đi cùng lúc
        future_to_drug = {executor.submit(fetch_single_drug_record, drug): drug for drug in drug_names}
        
        # Gom kết quả ngay khi luồng nào đó tải xong
        for future in concurrent.futures.as_completed(future_to_drug):
            drug_name, record, error = future.result()
            results[drug_name] = {"record": record, "error": error}
            
    return results

def get_field(record, possible_keys):
    """Quét các trường dự phòng."""
    for key in possible_keys:
        if key in record and record[key]:
            return record[key][0]
    return "No data available in the drug label."


# =====================================================================
# TASK 1: KIỂM TRA TƯƠNG TÁC CHO DANH SÁCH THUỐC
# =====================================================================
def get_fda_drug_interactions(drug_names: list) -> str:
    """
    Nhận mảng tên thuốc, tải song song và gom nhóm toàn bộ dữ liệu cảnh báo 
    để LLM (Agent) đối chiếu tương tác chéo.
    """
    fetched_data = fetch_multiple_drug_records(drug_names)
    combined_report = []
    
    target_fields = [
        'boxed_warning', 'contraindications', 'drug_interactions', 
        'warnings_and_cautions', 'precautions', 'pharmacokinetics', 
        'information_for_patients', 'description', "warnings"
    ]
    
    for drug in drug_names:
        data = fetched_data[drug]
        combined_report.append(f"========== [ {drug.upper()} ] ==========")
        
        if data["error"]:
            combined_report.append(data["error"] + "\n")
            continue
            
        record = data["record"]
        extracted_sections = []
        
        for field in target_fields:
            if field in record and record[field]:
                text_content = record[field][0]
                extracted_sections.append(f"[{field.upper()} SECTION]:\n{text_content}")
                if field == 'drug_interactions':
                    break # Dừng quét nếu đã trúng trường chuẩn
                    
        if extracted_sections:
            combined_report.append("\n\n".join(extracted_sections) + "\n")
        else:
            combined_report.append(f"Warning data sections could not be structured for {drug}.\n")
            
    return "\n\n".join(combined_report)


# =====================================================================
# TASK 2: HỒ SƠ AN TOÀN TRÊN BỆNH NHÂN CHO DANH SÁCH THUỐC
# =====================================================================
def format_single_profile(drug_name, profile):
    """Format json của 1 thuốc thành Text."""
    if "error" in profile:
        return f"--- {drug_name.upper()} ---\n{profile['error']}\n"

    report = f"--- COMPREHENSIVE SAFETY PROFILE FOR {drug_name.upper()} ---\n"
    report += f"- Indications: {profile['basic_information']['indications_and_usage']}\n"
    report += f"- Dosage: {profile['basic_information']['dosage_and_administration']}\n"
    report += f"- Boxed Warning: {profile['high_risk_warnings']['boxed_warning']}\n"
    report += f"- Contraindications: {profile['high_risk_warnings']['contraindications']}\n"
    report += f"- Adverse Reactions: {profile['side_effects_and_warnings']['adverse_reactions']}\n"
    report += f"- Pregnancy & Nursing: {profile['specific_populations']['pregnancy_and_nursing_mothers']}\n"
    report += f"- Geriatric Use: {profile['specific_populations']['geriatric_use']}\n"
    report += f"- Pediatric Use: {profile['specific_populations']['pediatric_use']}\n"
    return report

def get_fda_drug_safety_profile(drug_names: list, parse_to_text: bool = True):
    """
    Trích xuất hồ sơ an toàn cho toàn bộ mảng thuốc.
    Trả về Text báo cáo gộp HOẶC Dictionary (nếu parse_to_text=False).
    """
    fetched_data = fetch_multiple_drug_records(drug_names)
    final_profiles_json = {}
    final_profiles_text = []
    
    for drug in drug_names:
        data = fetched_data[drug]
        
        if data["error"]:
            final_profiles_json[drug] = {"error": data["error"]}
            final_profiles_text.append(f"--- {drug.upper()} ---\n{data['error']}\n")
            continue
            
        record = data["record"]
        profile = {
            "basic_information": {
                "indications_and_usage": get_field(record, ['indications_and_usage']),
                "dosage_and_administration": get_field(record, ['dosage_and_administration'])
            },
            "high_risk_warnings": {
                "boxed_warning": get_field(record, ['boxed_warning']),
                "contraindications": get_field(record, ['contraindications'])
            },
            "side_effects_and_warnings": {
                "general_warnings": get_field(record, ['warnings_and_cautions', 'precautions']), 
                "adverse_reactions": get_field(record, ['adverse_reactions'])
            },
            "specific_populations": {
                "pregnancy_and_nursing_mothers": get_field(record, ['pregnancy', 'nursing_mothers', 'use_in_specific_populations']),
                "geriatric_use": get_field(record, ['geriatric_use']),
                "pediatric_use": get_field(record, ['pediatric_use'])
            }
        }
        
        final_profiles_json[drug] = profile
        final_profiles_text.append(format_single_profile(drug, profile))
        
    if parse_to_text:
        return "\n\n=========================================\n\n".join(final_profiles_text)
    return final_profiles_json


# ---------------------------------------------------------------------
# TEST CHẠY THỬ MẢNG THUỐC
# ---------------------------------------------------------------------
if __name__ == "__main__":
    test_drugs = ["ibuprofen", "citalopram", "amoxicillin"]
    output_filename = "fda_multi_results.txt"
    
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write("--- MULTI-DRUG INTERACTION REPORT ---\n")
        # Gọi tool với input là mảng
        interactions = get_fda_drug_interactions(test_drugs)
        file.write(interactions + "\n\n")
        
        file.write("--- MULTI-DRUG SAFETY PROFILES ---\n")
        # Đổi tên biến thành profiles_text cho dễ hiểu và gọi parse_to_text=True
        profiles_text = get_drug_safety_profile(test_drugs, parse_to_text=True)
        
        # Ghi trực tiếp chuỗi văn bản vào file
        file.write(profiles_text + "\n")
        
    print(f"✅ Đã tải xong dữ liệu cho {len(test_drugs)} thuốc. Lưu tại: {output_filename}")