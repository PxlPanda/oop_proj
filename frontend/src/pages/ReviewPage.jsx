import { useParams } from 'react-router-dom';
import { useState, useEffect } from 'react';
import EditorModal from '../components/EditorModal';
import { getSymbols, updateSymbol } from '../api/api';

export default function ReviewPage() {
  const { sessionId } = useParams();

  const [symbols, setSymbols] = useState([]);
  const [imageUrl, setImageUrl] = useState('');
  const [editing, setEditing] = useState(null); // символ для редактирования

  useEffect(() => {
    async function fetchSymbols() {
      console.log('Загрузка символов для сессии:', sessionId);
      try {
        const response = await getSymbols(sessionId);
        console.log('✅ Ответ от сервера:', response.data);
        setSymbols(response.data.symbols);
        setImageUrl(response.data.image_url); // если backend отдаёт image_url
      } catch (err) {
        console.error('❌ Ошибка при загрузке символов:', err);
      }
    }
  
    fetchSymbols();
  }, [sessionId]);
  

  const handleEdit = (symbol) => {
    setEditing(symbol);
  };

  const handleSave = async (newChar) => {
    await updateSymbol(sessionId, editing.id, { char: newChar });
    setSymbols(prev =>
      prev.map(s =>
        s.id === editing.id ? { ...s, char: newChar } : s
      )
    );
    setEditing(null);
  };

  return (
    <div className="relative bg-gray-50 min-h-screen p-4">
      <h1 className="text-xl font-bold mb-4">Сессия: {sessionId}</h1>

      {imageUrl && (
        <div className="relative inline-block">
          <img src={imageUrl} alt="template" className="max-w-full" />

          {symbols.map((sym) => (
            <div
              key={sym.id}
              onClick={() => handleEdit(sym)}
              className="absolute bg-white bg-opacity-80 text-sm border border-blue-500 text-blue-800 rounded cursor-pointer flex items-center justify-center"
              style={{
                left: sym.x,
                top: sym.y,
                width: sym.width,
                height: sym.height,
                position: 'absolute',
              }}
            >
              {sym.char}
            </div>
          ))}
        </div>
      )}

      <EditorModal
        isOpen={!!editing}
        onClose={() => setEditing(null)}
        value={editing?.char || ''}
        onSave={handleSave}
      />
    </div>
  );
}
