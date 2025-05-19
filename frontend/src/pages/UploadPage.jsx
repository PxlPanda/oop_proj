// src/pages/UploadPage.jsx
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadTemplate } from '../api/api';

export default function UploadPage() {
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

  const handleSubmit = async () => {
    if (!file) {
      alert('Пожалуйста, выбери файл.');
      return;
    }

    setLoading(true);
    try {
      const response = await uploadTemplate(file);
      const sessionId = response.data.sessionId;
      navigate(`/review/${sessionId}`);
    } catch (err) {
      console.error(err);
      alert('Ошибка при загрузке шаблона');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6 flex flex-col items-center">
      <h1 className="text-2xl font-bold mb-4">Загрузка шаблона</h1>

      <input
        type="file"
        accept="image/*"
        onChange={handleChange}
        className="mb-4"
      />

      {preview && (
        <img
          src={preview}
          alt="preview"
          className="max-w-md border rounded shadow mb-4"
        />
      )}

      <button
        onClick={handleSubmit}
        disabled={loading}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
      >
        {loading ? 'Загрузка...' : 'Загрузить'}
      </button>
    </div>
  );
}
