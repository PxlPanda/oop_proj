import { useState, useEffect } from 'react';

export default function EditorModal({ isOpen, onClose, value, onSave }) {
  const [edited, setEdited] = useState(value);

  // Чтобы при открытии модалки поле заполнялось текущим символом
  useEffect(() => {
    setEdited(value);
  }, [value]);

  if (!isOpen) return null;

  const handleSave = () => {
    onSave(edited);
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 shadow-md w-80">
        <h2 className="text-lg font-semibold mb-4">Редактировать символ</h2>
        <input
          type="text"
          maxLength={1}
          value={edited}
          onChange={(e) => setEdited(e.target.value)}
          className="w-full border px-3 py-2 mb-4 rounded text-center text-xl"
          autoFocus
        />
        <div className="flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300"
          >
            Отмена
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Сохранить
          </button>
        </div>
      </div>
    </div>
  );
}
