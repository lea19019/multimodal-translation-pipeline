import React, { useState } from 'react';
import { Type, Send } from 'lucide-react';

const TextInput = ({ onTextSubmit, disabled = false }) => {
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim() && !disabled) {
      onTextSubmit(text);
    }
  };

  return (
    <div className="space-y-4">
      <form onSubmit={handleSubmit}>
        <div className="space-y-4">
          <div>
            <label htmlFor="text-input" className="block text-sm font-medium text-gray-700 mb-2">
              Enter text to translate
            </label>
            <div className="relative">
              <Type className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
              <textarea
                id="text-input"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Type your text here..."
                className="block w-full pl-10 pr-3 py-3 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500 resize-none"
                rows={4}
                disabled={disabled}
              />
            </div>
          </div>
          
          <div className="flex justify-end">
            <button
              type="submit"
              disabled={!text.trim() || disabled}
              className="inline-flex items-center gap-2 px-6 py-2 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send size={16} />
              Translate
            </button>
          </div>
        </div>
      </form>
      
      {text && (
        <div className="mt-4 p-3 bg-gray-50 rounded-md">
          <p className="text-sm text-gray-600">
            <strong>Character count:</strong> {text.length}
          </p>
        </div>
      )}
    </div>
  );
};

export default TextInput;