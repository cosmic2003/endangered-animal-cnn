import { useState } from 'react';
import UploadForm from './components/UploadForm';
import AnimalInfo from './components/AnimalInfo';
import './App.css';

const IUCN_GRADES = [
  { code: 'EX', name: '절멸',      color: '#3a3a3a', desc: '야생에서 완전히 사라진 종' },
  { code: 'EW', name: '야생절멸',  color: '#6b6b6b', desc: '야생에서만 절멸, 사육시설에 생존' },
  { code: 'CR', name: '위급',      color: '#c1121f', desc: '극도로 높은 절멸 위기' },
  { code: 'EN', name: '위기',      color: '#e76f51', desc: '매우 높은 절멸 위기' },
  { code: 'VU', name: '취약',      color: '#d4a017', desc: '높은 절멸 위기' },
  { code: 'NT', name: '준위협',    color: '#52796f', desc: '가까운 미래에 위협 가능' },
  { code: 'LC', name: '관심 대상', color: '#2d6a4f', desc: '현재 절멸 위험 낮음' },
];

function App() {
  const [animalInfo, setAnimalInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);

  const handleClassification = (file) => {
    setIsLoading(true);
    setAnimalInfo(null);

    const reader = new FileReader();
    reader.onloadend = () => setImagePreview(reader.result);
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append('file', file);

    fetch('http://localhost:8000/classify/', {
      method: 'POST',
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        setAnimalInfo(data);
        setIsLoading(false);
      })
      .catch((err) => {
        console.error('Error:', err);
        setIsLoading(false);
      });
  };

  const handleReset = () => {
    setImagePreview(null);
    setAnimalInfo(null);
  };

  return (
    <div className="app">

      {/* ── 헤더 ── */}
      <header className="app-header">
        <span className="leaf-icon">🌿</span>
        <h1>멸종위기 동물 분류 서비스</h1>
        <p className="subtitle">AI 기반 야생동물 보호 프로젝트</p>
      </header>

      {/* ── 히어로 배너 ── */}
      <section className="hero-banner">
        <div className="hero-deco" aria-hidden="true">
          <span>🐯</span><span>🦏</span><span>🐘</span><span>🦁</span>
          <span>🐼</span><span>🦅</span><span>🐢</span><span>🦋</span>
        </div>
        <h2 className="hero-title">지구의 동물들을 기록합니다</h2>
        <p className="hero-desc">
          동물 사진 한 장으로 멸종위기 여부와 IUCN 보호 등급을 확인하세요.<br />
          우리의 기록이 자연을 지키는 첫걸음이 됩니다.
        </p>
      </section>

      <main className="app-main">

        {/* ── 사용 방법 ── */}
        <section className="steps-section">
          <h3 className="section-title">사용 방법</h3>
          <div className="steps-grid">
            <div className="step-card">
              <span className="step-emoji">📸</span>
              <h4>사진 업로드</h4>
              <p>동물 사진을 선택하거나<br />끌어다 놓으세요</p>
            </div>
            <div className="step-arrow">›</div>
            <div className="step-card">
              <span className="step-emoji">🤖</span>
              <h4>AI 분석</h4>
              <p>딥러닝 모델이<br />동물의 종을 자동 판별합니다</p>
            </div>
            <div className="step-arrow">›</div>
            <div className="step-card">
              <span className="step-emoji">🌿</span>
              <h4>결과 확인</h4>
              <p>IUCN 보호 등급과<br />생태 정보를 확인하세요</p>
            </div>
          </div>
        </section>

        {/* ── 업로드 카드 ── */}
        <div className="upload-card">
          <UploadForm
            onClassification={handleClassification}
            imagePreview={imagePreview}
          />
          {imagePreview && (
            <button className="reselect-btn" onClick={handleReset}>
              다른 사진 선택
            </button>
          )}
          {isLoading && (
            <div className="loading-box">
              <div className="loading-spinner" />
              <p>AI가 동물을 분석하고 있습니다...</p>
            </div>
          )}
        </div>

        {/* ── 분류 결과 ── */}
        {animalInfo && <AnimalInfo animalInfo={animalInfo} />}

        {/* ── IUCN 등급 안내 ── */}
        <section className="iucn-guide">
          <h3 className="section-title">IUCN 국제 보호 등급이란?</h3>
          <p className="iucn-guide-intro">
            국제자연보전연맹(IUCN)이 전 세계 생물종의 절멸 위험도를 평가하는 기준입니다.
          </p>
          <div className="iucn-grade-list">
            {IUCN_GRADES.map(({ code, name, color, desc }) => (
              <div key={code} className="iucn-grade-item">
                <span className="iucn-grade-badge" style={{ backgroundColor: color }}>
                  {code}
                </span>
                <div className="iucn-grade-text">
                  <strong>{name}</strong>
                  <span>{desc}</span>
                </div>
              </div>
            ))}
          </div>
        </section>

      </main>

      <footer className="app-footer">
        © 2025 멸종위기 동물 분류 서비스 — 자연을 지키는 기술
      </footer>
    </div>
  );
}

export default App;
