// src/components/ReviewToggle.jsx
import React from 'react';

export default function ReviewToggle({ isReviewed, onToggle }) {
  return (
    <div className="my-4 text-center">
      <button
        onClick={onToggle}
        className={`px-4 py-2 rounded font-semibold transition-colors duration-200 
          ${isReviewed ? 'bg-green-500 text-white hover:bg-green-600' : 'bg-gray-300 text-black hover:bg-gray-400'}`}
      >
        {isReviewed ? '✅ Отметить как не проверенный' : '🔍 Отметить как проверенный'}
      </button>
    </div>
  );
}