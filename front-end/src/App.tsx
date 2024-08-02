import React from 'react';
import ChatBotUI from './ChatBotUI';
import { DataProvider } from './DataContext';

const App: React.FC = () => (
  <DataProvider>
    <ChatBotUI />
  </DataProvider>
);

export default App;
