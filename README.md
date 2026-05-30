# Folder findings and impressions
## Nội dung
- Chứa source code preprocess bộ CT RATE và code train models.
- Phần others là của bộ CVD. Chưa xong phần tiền xử lý
## Cấu trúc dataset để sẵn sàng cho finetune
### Preprocess
#### Ảnh
- HU -> RGB -> Resize. Định dạng (Slices, H, W, C)
#### Text
- Tạo prompt đầu vào và đầu ra
#### Link Dataset [https://huggingface.co/datasets/rimine/ct-rate-medgemma-ready]

## Lưu ý khi finetune
- Bật flash attention thì chậm hơn
- batchsize_per_device bằng 2 thì tối ưu nhất
- Cấu hình khi chạy bằng modal: B100:4CPU:128GB
- Steps: 125(500 samples) thì chạy tầm 1 tiếng rưỡi.



