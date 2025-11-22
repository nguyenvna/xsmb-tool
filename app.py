import streamlit as st
import pandas as pd
from collections import Counter
import random
import re

# 1. Cấu hình trang web
st.set_page_config(page_title="Trợ Lý Thần Tài XSMB", page_icon="💰")

st.title("💰 Trợ Lý Dự Đoán XSMB - AI Analytics")
st.write("Nhập dữ liệu giải đặc biệt 30 ngày qua để hệ thống phân tích.")

# 2. Sidebar nhập liệu
with st.sidebar:
    st.header("Dữ liệu đầu vào")
    input_method = st.radio("Chọn cách nhập:", ["Dán dữ liệu (Copy/Paste)", "Dùng dữ liệu mẫu"])
    
    raw_data = ""
    if input_method == "Dán dữ liệu (Copy/Paste)":
        raw_data = st.text_area("Dán cột Giải đặc biệt vào đây:", height=300)
    else:
        # Dữ liệu mẫu giả lập
        raw_data = """58293\n10234\n59188\n32099\n11245\n99821\n45678\n12345\n67890\n13579\n24680\n11111\n22222\n33333\n44444\n55555\n66666\n77777\n88888\n99999\n12121\n34343\n56565\n78787\n90909\n12312\n45645\n78978\n32132\n65465"""
        st.info("Đã nạp dữ liệu mẫu.")

    btn_analyze = st.button("🚀 Phân tích ngay")

# 3. Xử lý dữ liệu
def process_data(text_data):
    # Lọc bỏ ký tự lạ, chỉ lấy số, mỗi dòng 1 số
    lines = text_data.strip().split('\n')
    clean_data = []
    for line in lines:
        nums = re.findall(r'\d{5}', line) # Tìm chuỗi 5 số
        if nums:
            clean_data.extend(nums)
    return clean_data

# 4. Giao diện chính
if btn_analyze and raw_data:
    history = process_data(raw_data)
    
    if len(history) < 5:
        st.error("Dữ liệu quá ít hoặc không đúng định dạng 5 số. Vui lòng kiểm tra lại.")
    else:
        st.success(f"Đã nhận diện {len(history)} ngày kết quả.")
        
        # Tách số
        de_list = [x[-2:] for x in history]
        ba_cang_list = [x[-3:] for x in history]
        
        # --- PHÂN TÍCH ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Thống kê Đề (2 số cuối)")
            count_de = Counter(de_list)
            top_de = count_de.most_common(5)
            df_de = pd.DataFrame(top_de, columns=['Số', 'Số lần về'])
            st.dataframe(df_de, use_container_width=True)
            
        with col2:
            st.subheader("🔥 Dự đoán Đề (Top 10)")
            # Logic dự đoán đơn giản: Top hay về + Ngẫu nhiên có trọng số
            predictions = [x[0] for x in top_de]
            while len(predictions) < 10:
                new_num = f"{random.randint(0,99):02d}"
                if new_num not in predictions:
                    predictions.append(new_num)
            
            st.write(", ".join(predictions))
            
        st.markdown("---")
        
        st.subheader("🔮 Dự đoán 3 Càng (Tham khảo)")
        # Logic ghép càng giả lập
        cang_du_doan = []
        for de in predictions[:5]: # Lấy 5 số đề mạnh nhất
             cang = random.randint(0, 9)
             cang_du_doan.append(f"{cang}{de}")
        
        # Thêm 5 số ngẫu nhiên
        while len(cang_du_doan) < 10:
             cang_du_doan.append(f"{random.randint(0,999):03d}")
             
        st.success(", ".join(cang_du_doan))

        st.warning("⚠️ Lưu ý: Kết quả chỉ mang tính chất tham khảo giải trí. Chúc bạn may mắn!")

else:
    if not raw_data and btn_analyze:
        st.warning("Vui lòng nhập dữ liệu trước khi phân tích.")
