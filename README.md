
# Đối với file speech-emotion-recognition.ipynb
## Tải lên Kaggle
Để có thể chạy file speech-emotion-recognition.ipynb, bạn cần tải nó lên Kaggle. Để làm điều này, hãy thực hiện các bước sau:

Truy cập trang web [Kaggle](https://www.kaggle.com/).
-  Nhấp vào nút Create.
-  Chọn New Notebook.
-  Nhấp vào File
-  Nhấp vào nút Import Notebook.
-  Chọn tệp speech_emotion_recognition.ipynb.
-  Nhấp vào nút Import.
Sau khi tệp đã được tải lên, bạn có thể truy cập nó từ Kaggle.

## Thêm dữ liệu

Để có thể chạy mô hình nhận dạng cảm xúc giọng nói, bạn cần thêm các tập dữ liệu sau:

-  RAVDESS: Tập dữ liệu âm thanh biểu cảm giọng nói.
-  CREMA-D: Tập dữ liệu âm thanh biểu cảm giọng nói trong các tình huống thực tế.
-  TESS: Tập dữ liệu âm thanh biểu cảm giọng nói trong các tình huống giao tiếp.
Để thêm các tập dữ liệu này, hãy thực hiện các bước sau:

-  Trong Notebook,ở bên phải có tùy chọn Data, chọn Add data
-  Nhấp vào thanh Search.
-  Nhập tên tập dữ liệu bạn muốn thêm.
-  Nhấp vào nút Search.
-  Nhấp vào nút Add.
Sau khi các tập dữ liệu đã được thêm vào dự án, bạn có thể ấn Run,chọn Run All để chạy toàn bộ notebook.

## đối với file Data_collection.ipynb, Speech-emotion-recognition-train.ipynb
-  2 file này được tách đôi từ speech-emotion-recognition.ipynb :
-    Data_collection.ipynb dùng để tạo ra file Emotion.csv
-    speech-emotion-recognition-train.ipynb sử dụng trực tiếp file Emotion.csv để huấn luyện mô hình
   (chủ yếu là CNN model-mô hình tốt nhất để lấy lại file encoder, file weight phục vụ cho việc inferience)

## Lưu ý

Bạn cần có tài khoản Kaggle để tải lên và truy cập các tập dữ liệu.
Bạn cần có Python 3 và các thư viện sau để chạy mô hình:
-  numpy
-  pandas
-  scikit-learn
-  librosa
-  tensorflow
-  keras

# Đối với file inference.py
- Cần có các file best-model-local-weight.h5 , encoder.pkl, scaler.pkl
- Để chạy, cần máy tính có GPU đủ mạnh
- Để thử với các file âm thanh khác (định dạng wav), cần có đường dẫn của file, cũng như nhãn cảm xúc được xác định sẵn ( dùng để test mô hình)


#Chúc bạn thành công!
