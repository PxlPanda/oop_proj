import { useParams, useNavigate } from 'react-router-dom';
import { useState } from 'react';
import EditorModal from '../components/EditorModal';

export default function ReviewPage() {
  const { sessionId } = useParams();
  const navigate = useNavigate();

  // Список символов: по умолчанию — символ A, id от 1 до 8
  const [symbols, setSymbols] = useState(
    Array.from({ length: 8 }, (_, i) => ({
      id: i + 1,
      char: 'A',
    }))
  );

  const [editing, setEditing] = useState(null); // { id, char } или null

  const handleEdit = (symbol) => {
    setEditing(symbol);
  };

  const handleSave = (newChar) => {
    setSymbols((prev) =>
      prev.map((s) =>
        s.id === editing.id ? { ...s, char: newChar } : s
      )
    );
  };

  const handleMarkReviewed = () => {
    alert('Отметка как проверено (заглушка)');
  };

  const handleExport = () => {
    const json = JSON.stringify(symbols, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `symbols_session_${sessionId}.json`;
    a.click();
  };

  return (
    <div className="min-h-screen bg-white p-6">
      <h1 className="text-2xl font-bold mb-4">
        Проверка символов (сессия: <span className="text-blue-600">{sessionId}</span>)
      </h1>

      {/* Сетка карточек */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        {symbols.map((symbol) => (
          <div
            key={symbol.id}
            className="border rounded p-4 text-center shadow hover:shadow-md transition cursor-pointer"
            onClick={() => handleEdit(symbol)}
          >
            <p className="text-2xl font-mono">{symbol.char}</p>
            <p className="text-sm text-gray-500">ID: {symbol.id}</p>
          </div>
        ))}
      </div>

      {/* Кнопки */}
      <div className="flex flex-wrap gap-4">
        <button
          onClick={handleMarkReviewed}
          className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
        >
          ✅ Отметить как проверено
        </button>
        <button
          onClick={handleExport}
          className="px-4 py-2 bg-yellow-500 text-white rounded hover:bg-yellow-600"
        >
          ⬇️ Скачать JSON
        </button>
        <button
          onClick={() => navigate('/')}
          className="px-4 py-2 bg-gray-300 text-black rounded hover:bg-gray-400"
        >
          ⬅️ Назад
        </button>
      </div>

      {/* Модалка правки */}
      <EditorModal
        isOpen={!!editing}
        onClose={() => setEditing(null)}
        value={editing?.char || ''}
        onSave={handleSave}
      />
    </div>
  );
}
