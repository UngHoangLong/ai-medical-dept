# Git Workflow — Hướng dẫn làm việc nhóm

## Quy tắc đặt tên nhánh

```
<tên_thành_viên>/<chức_năng>

Ví dụ:
  ung_long/agents/data_prep
  truong/training/radiology
  tran/retrieval/faiss-index
  minh/serving/lora-manager
```

---

## Luồng làm việc chuẩn

### 1. Bắt đầu làm việc (mỗi khi mở repo lên)

Luôn pull code mới nhất từ `dev` về trước khi code:

```bash
git checkout dev
git pull origin dev
```

### 2. Tạo nhánh mới cho chức năng

```bash
git checkout -b <tên>/<chức_năng>

# Ví dụ:
git checkout -b ung_long/agents/data_prep
```

### 3. Code, add, commit

```bash
git add .                             # hoặc git add <file> nếu chỉ muốn add một số file
git commit -m "mô tả ngắn thay đổi"
```

Lưu ý không được push .env, đảm bảo phải cập nhật các thư mục, file nhạy cảm trong .gitignore

### 4. Trước khi push — đồng bộ code mới nhất từ dev

Nếu trong lúc bạn code, người khác đã merge vào `dev`, cần rebase để tránh conflict lúc tạo PR:

```bash
git fetch origin
git rebase origin/dev
```

Nếu có conflict thì fix từng file, sau đó:

```bash
git add <file_đã_fix>
git rebase --continue
```

### 5. Push nhánh lên GitHub

Lần đầu push nhánh mới:

```bash
git push --set-upstream origin <tên>/<chức_năng>
```

Các lần sau:

```bash
git push
```

---

## Tạo Pull Request lên `dev`

1. Vào GitHub → repo `ai-medical-dept`
2. Sẽ thấy banner vàng **"Compare & pull request"** → bấm vào
3. Kiểm tra:
   - **base**: `dev`
   - **compare**: nhánh của bạn
4. Điền tiêu đề và mô tả những gì đã làm
5. Bấm **"Create pull request"**

---

## Review code (cho Reviewer)

Repo yêu cầu **ít nhất 2 reviewer** approve trước khi merge.

### Cách review trên GitHub:

1. Vào tab **"Pull requests"** → chọn PR cần review
2. Bấm tab **"Files changed"** để xem toàn bộ thay đổi
3. Hover vào dòng code muốn comment → bấm dấu **`+`** xuất hiện bên trái
4. Viết comment → chọn:
   - **"Add single comment"** — comment đơn lẻ
   - **"Start a review"** — gom nhiều comment lại, gửi 1 lần
5. Sau khi xem hết, bấm **"Review changes"** (góc phải trên) → chọn:
   - **Comment** — chỉ nhận xét, chưa approve
   - **Approve** — đồng ý merge
   - **Request changes** — yêu cầu sửa trước khi merge
6. Bấm **"Submit review"**

### Khi bị Request changes:

```bash
# Sửa code theo góp ý
git add <file>
git commit -m "fix: ..."
git push
# PR tự động cập nhật, không cần tạo PR mới
```

### Resolve conversation:

Sau khi đã sửa theo comment của reviewer, bấm **"Resolve conversation"** trên từng comment để đánh dấu đã xử lý xong.

---

## Merge PR vào `dev`

Khi đủ 2 Approve và tất cả conversation đã Resolve:

1. Bấm **"Squash and merge"** (gom tất cả commit thành 1)
2. Sửa commit message cho rõ ràng nếu cần
3. Bấm **"Confirm squash and merge"**
4. Xóa nhánh sau khi merge: bấm **"Delete branch"**

---

## Tóm tắt nhanh

```
pull dev → tạo nhánh → code → rebase dev → push → tạo PR → review → merge
```
