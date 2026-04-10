import React, { useState } from 'react';
import UploadForm from './components/UploadForm';
import AnimalInfo from './components/AnimalInfo';
import './App.css';

function App() {
  const [animalInfo, setAnimalInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(false); // 로딩 상태 추가

  const handleClassification = (file) => {
    setIsLoading(true); // 로딩 시작
    setAnimalInfo(null); // 이전 결과 초기화

    // 1. 파일을 담을 빈 상자(FormData)를 만듭니다.
    const formData = new FormData();
    
    // 2. 상자에 'file'이라는 이름표를 붙여서 사진을 넣습니다. 
    // (이름이 백엔드의 매개변수 이름과 꼭 같아야 합니다!)
    formData.append('file', file);

    // 3. 백엔드 API로 상자를 통째로 보냅니다.
    fetch('http://localhost:8000/classify/', {
      method: 'POST',
      // 주의: FormData를 보낼 때는 headers에 'Content-Type'을 적지 않습니다! 브라우저가 알아서 처리해 줍니다.
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      setAnimalInfo(data); // 받은 데이터를 저장
      setIsLoading(false); // 로딩 끝
    })
    .catch(error => {
      console.error('Error:', error);
      setIsLoading(false); // 에러가 나도 로딩은 끝내기
    });
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>멸종위기 동물 판독기 🐾</h1>
        <UploadForm onClassification={handleClassification} />
        {/* 로딩 중일 때 보여줄 문구 */}
        {isLoading && <p>AI가 동물을 분석하고 있습니다...</p>}
      </header>
      {/* 데이터가 성공적으로 들어오면 AnimalInfo 컴포넌트를 보여줍니다 */}
      {animalInfo && <AnimalInfo animalInfo={animalInfo} />}
    </div>
  );
}

export default App;
