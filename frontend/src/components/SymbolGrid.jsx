// src/components/SymbolGrid.jsx
import React from 'react';

export default function SymbolGrid({ symbols, onEdit }) {
  return (
    <div className="grid grid-cols-6 gap-4">
      {symbols.map((symbol) => (
        <div
          key={symbol.id}
          className="relative border p-2 rounded shadow hover:ring-2 hover:ring-blue-400"
        >
          <img
            src={symbol.image_path}
            alt={symbol.label}
            className="w-full h-auto object-contain mb-1"
          />
          <div className="text-sm text-center">
            <strong>{symbol.label}</strong>
            {symbol.predicted && symbol.predicted !== symbol.label && (
              <span className="text-red-500"> (предсказано: {symbol.predicted})</span>
            )}
          </div>
          <button
            onClick={() => onEdit(symbol)}
            className="absolute top-1 right-1 text-xs bg-yellow-300 px-1 rounded"
          >
            Править
          </button>
        </div>
      ))}
    </div>
  );
}