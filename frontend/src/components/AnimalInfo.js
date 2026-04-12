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
  if (animalInfo.error) {
    return (
      <div className="error-card">
        <h3>분석 오류</h3>
        <p>{animalInfo.error}</p>
      </div>
    );
  }

  const code = getStatusCode(animalInfo.iucn_status);
  const badgeColor = code ? IUCN_COLORS[code] : '#52796f';
  const badgeLabel = code ? IUCN_LABELS[code] : animalInfo.iucn_status;
  const icon = ANIMAL_ICONS[code] || ANIMAL_ICONS.default;

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
        <div className="iucn-row">
          <span className="iucn-label">IUCN 보호 등급</span>
          <span className="iucn-badge" style={{ backgroundColor: badgeColor }}>
            {badgeLabel}
          </span>
        </div>

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
