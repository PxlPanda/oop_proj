// src/pages/UploadPage.jsx
import { useState } from 'react';

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
    }
  };

  const handleSubmit = () => {
    alert('Пока что просто заглушка. Загрузим позже в API.');
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
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
      >
        Загрузить
      </button>
    </div>
  );
}
