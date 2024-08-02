import React, { createContext, useContext, useState, ReactNode } from 'react';

interface DataContextProps {
  fetchedData: any;
  setFetchedData: (data: any) => void;
  emailaddress: string;
  setEmailAddress: (data: string) => void;
}

const DataContext = createContext<DataContextProps | undefined>(undefined);

export const DataProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [fetchedData, setFetchedData] = useState<any>(null);
  const [emailaddress, setEmailAddress] = useState<string>('');

  return (
    <DataContext.Provider value={{ fetchedData, setFetchedData, emailaddress, setEmailAddress }}>
      {children}
    </DataContext.Provider>
  );
};

export const useDataContext = (): DataContextProps => {
  const context = useContext(DataContext);
  if (!context) {
    throw new Error('useDataContext must be used within a DataProvider');
  }
  return context;
};
