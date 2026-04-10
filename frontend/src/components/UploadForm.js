import React from 'react';

function UploadForm({ onClassification }) {
  const handleFileChange = (event) => {
    // 사용자가 선택한 실제 파일 객체를 가져옵니다.
    const file = event.target.files[0];
    if (file) {
      // 파일 원본을 그대로 App.js로 올려보냅니다.
      onClassification(file);
    }
  };

  return (
    <div className="UploadForm">
      <input type="file" accept="image/*" onChange={handleFileChange} />
    </div>
  );
}

export default UploadForm;