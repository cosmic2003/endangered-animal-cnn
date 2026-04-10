// frontend/src/components/AnimalInfo.js
import React from 'react';

function AnimalInfo({ animalInfo }) {
  // 만약 데이터에 에러 메시지가 있다면 에러를 보여줍니다.
  if (animalInfo.error) {
    return (
      <div className="AnimalInfo">
        <h2 style={{ color: 'red' }}>오류 발생!</h2>
        <p>{animalInfo.error}</p>
      </div>
    );
  }

  // 파이썬 백엔드에서 보낸 이름(animal_name, iucn_status 등)과 똑같이 맞춰서 출력합니다.
  return (
    <div className="AnimalInfo" style={{ marginTop: '20px', padding: '20px', border: '1px solid #ddd', borderRadius: '8px' }}>
      <h2>분류 결과: {animalInfo.animal_name} 🐯</h2>
      <p><strong>학명:</strong> {animalInfo.scientific_name}</p>
      <p><strong>IUCN 등급:</strong> <span style={{ color: '#d9534f', fontWeight: 'bold' }}>{animalInfo.iucn_status}</span></p>
      <p><strong>설명:</strong> {animalInfo.description}</p>
    </div>
  );
}

export default AnimalInfo;