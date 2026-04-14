import { useEffect, useState } from 'react';

const IUCN_COLORS = {
  EX: '#3a3a3a',
  EW: '#6b6b6b',
  CR: '#c1121f',
  EN: '#e76f51',
  VU: '#d4a017',
  NT: '#52796f',
  LC: '#2d6a4f',
};

const IUCN_LABELS = {
  EX: '절멸 (EX)',
  EW: '야생절멸 (EW)',
  CR: '위급 (CR)',
  EN: '위기 (EN)',
  VU: '취약 (VU)',
  NT: '준위협 (NT)',
  LC: '관심 대상 (LC)',
};

const ANIMAL_ICONS = {
  default: '🐾',
  CR: '🚨',
  EN: '⚠️',
  VU: '🔶',
  NT: '🌿',
  LC: '✅',
};

function getStatusCode(status) {
  if (!status) return null;
  const upper = status.toUpperCase();
  return Object.keys(IUCN_COLORS).find((key) => upper.includes(key)) || null;
}

function AnimalInfo({ animalInfo }) {
  const [barWidths, setBarWidths] = useState([0, 0, 0]);

  // 신뢰도 바 애니메이션
  useEffect(() => {
    setBarWidths([0, 0, 0]);
    const timer = setTimeout(() => {
      if (animalInfo.top3) {
        setBarWidths(animalInfo.top3.map((t) => t.confidence));
      }
    }, 100);
    return () => clearTimeout(timer);
  }, [animalInfo]);

  if (animalInfo.error) {
    return (
      <div className="error-card">
        <h3>분석 오류</h3>
        <p>{animalInfo.error}</p>
      </div>
    );
  }

  const code       = getStatusCode(animalInfo.iucn_status);
  const badgeColor = code ? IUCN_COLORS[code] : '#52796f';
  const badgeLabel = code ? IUCN_LABELS[code] : animalInfo.iucn_status;
  const icon       = ANIMAL_ICONS[code] || ANIMAL_ICONS.default;

  return (
    <div className="result-card">
      <div className="result-card-header">
        <span className="result-icon">{icon}</span>
        <div>
          <h2>{animalInfo.animal_name}</h2>
          {animalInfo.scientific_name && (
            <p className="scientific">{animalInfo.scientific_name}</p>
          )}
        </div>
      </div>

      <div className="result-card-body">

        {/* ── IUCN 등급 ── */}
        <div className="iucn-row">
          <span className="iucn-label">IUCN 보호 등급</span>
          <span className="iucn-badge" style={{ backgroundColor: badgeColor }}>
            {badgeLabel}
          </span>
        </div>

        {/* ── Top 3 신뢰도 바 ── */}
        {animalInfo.top3 && (
          <div className="top3-section">
            <p className="top3-title">AI 분석 후보</p>
            {animalInfo.top3.map((item, i) => (
              <div key={item.name} className="top3-row">
                <span className={`top3-rank ${i === 0 ? 'top3-rank--first' : ''}`}>
                  {i + 1}
                </span>
                <span className="top3-name">{item.name}</span>
                <div className="top3-bar-bg">
                  <div
                    className="top3-bar-fill"
                    style={{
                      width: `${barWidths[i]}%`,
                      backgroundColor: i === 0 ? badgeColor : '#74c69d',
                    }}
                  />
                </div>
                <span className="top3-pct">{item.confidence}%</span>
              </div>
            ))}
          </div>
        )}

        {/* ── Grad-CAM 히트맵 ── */}
        {animalInfo.gradcam && (
          <div className="gradcam-section">
            <p className="gradcam-title">AI 집중 분석 영역</p>
            <p className="gradcam-desc">빨간 영역일수록 AI가 중점적으로 본 부분이에요</p>
            <img
              src={`data:image/jpeg;base64,${animalInfo.gradcam}`}
              alt="Grad-CAM 히트맵"
              className="gradcam-img"
            />
          </div>
        )}

        {/* ── 설명 ── */}
        {animalInfo.description && (
          <div className="description-box">
            <p>{animalInfo.description}</p>
          </div>
        )}

      </div>
    </div>
  );
}

export default AnimalInfo;
