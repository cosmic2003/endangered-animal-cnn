import { useRef, useState } from 'react';

function UploadForm({ onClassification, imagePreview }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFile = (file) => {
    if (file && file.type.startsWith('image/')) {
      onClassification(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  return (
    <div
      className={`upload-zone ${isDragging ? 'dragging' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => !imagePreview && fileInputRef.current.click()}
    >
      {imagePreview ? (
        <img src={imagePreview} alt="업로드한 동물" className="preview-image" />
      ) : (
        <div className="upload-placeholder">
          <span className="upload-icon">🦁</span>
          <p>사진을 여기에 끌어다 놓거나</p>
          <span className="upload-btn-text">파일 선택하기</span>
          <span className="upload-hint">JPG, PNG, WEBP 지원</span>
        </div>
      )}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={(e) => handleFile(e.target.files[0])}
        style={{ display: 'none' }}
      />
    </div>
  );
}

export default UploadForm;
