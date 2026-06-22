export const getActivities = () => {
  try {
    const raw = localStorage.getItem('asc_activities');
    if (!raw || raw === 'null') return [];
    const list = JSON.parse(raw);
    return Array.isArray(list) ? list.filter(Boolean) : [];
  } catch (e) {
    return [];
  }
};

export const logActivity = (type, details, status = 'success') => {
  try {
    const list = getActivities();
    const newActivity = {
      id: Date.now(),
      type,
      details,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      status
    };
    const updated = [newActivity, ...list].slice(0, 10);
    localStorage.setItem('asc_activities', JSON.stringify(updated));
  } catch (e) {
    console.error('Failed to log activity', e);
  }
};

