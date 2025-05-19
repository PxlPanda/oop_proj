//Uploader.jsx
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadTemplate } from '../api/api';

export default function Uploader() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
    }
  };

  const handleUpload = async () => {
    if (!file) return alert("Выберите файл");
    setLoading(true);

    try {
      const response = await uploadTemplate(file);
      const sessionId = response.data.sessionId; // или как возвращает твой API
      navigate(`/review/${sessionId}`);
    } catch (err) {
      alert('Ошибка загрузки: ' + (err.response?.data?.message || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <input type="file" accept="image/*" onChange={handleChange} disabled={loading} />
      {preview && (
        <img
          src={preview}
          alt="preview"
          className="max-w-md border rounded shadow"
        />
      )}
      <button
        onClick={handleUpload}
        disabled={loading}
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
      >
        {loading ? 'Загрузка...' : 'Загрузить шаблон'}
      </button>
    </div>
  );
}
