import { useRef, useState } from 'react';

function UploadForm({ onClassification, imagePreview }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef   = useRef(null);
  const cameraInputRef = useRef(null);

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
    >
      {imagePreview ? (
        <img src={imagePreview} alt="업로드한 동물" className="preview-image" />
      ) : (
        <div className="upload-placeholder">
          <span className="upload-icon">🦁</span>
          <p>사진을 여기에 끌어다 놓거나</p>
          <div className="upload-btn-group">
            {/* label 방식 — iOS Safari에서도 안정적으로 동작 */}
            <label className="upload-btn upload-btn--file" htmlFor="file-input">
              파일 선택하기
            </label>
            <label className="upload-btn upload-btn--camera" htmlFor="camera-input">
              📷 카메라 촬영
            </label>
          </div>
          <span className="upload-hint">JPG, PNG, WEBP 지원</span>
        </div>
      )}

      {/* 파일 선택 */}
      <input
        id="file-input"
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={(e) => handleFile(e.target.files[0])}
        style={{ display: 'none' }}
      />
      {/* 카메라 촬영 (모바일에서 카메라 직접 실행) */}
      <input
        id="camera-input"
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={(e) => handleFile(e.target.files[0])}
        style={{ display: 'none' }}
      />
    </div>
  );
}

export default UploadForm;
