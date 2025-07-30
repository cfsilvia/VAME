import pandas as pd
import numpy as np
import os

class create_file_for_vame:
    def __init__(self,filename, sheetname, upper_tube, lower_tube):
        self._filename = filename
        self._upper_tube = upper_tube
        self._lower_tube = lower_tube
        self._sheetname = sheetname
        
    def read_excel(self):
        try:
           df = pd.read_excel(self._filename , sheet_name=self._sheetname) 
           return df 
        except Exception as e:
         print(f"An error occurred while reading the Excel file: {e}")
         return None
    '''
    Fill no detection since the blind mole wasn't inside the tube with zero only x and y
    ''' 
    def fill_no_detection(self,df):   
        updated_df = df.copy()
        for col in updated_df.columns:
            if '_x' in col.lower() or '_y' in col.lower():
                updated_df[col] = updated_df[col].replace(0, np.nan)
        return updated_df
    
    '''
    Remove the y which are outside the tube
    '''
    def remove_data_outside_tube(self,df):
        updated_df = df.copy()
        for col in updated_df.columns:
            if  '_y' in col.lower():
                # Replace values greater than the threshold with NaN
                updated_df[col] = df[col].mask(df[col] > self._lower_tube, np.nan)
                updated_df[col] = df[col].mask(df[col] < self._upper_tube, np.nan)
        return updated_df
    '''
    Remove the middle
    '''
    def  remove_middle(self,df): 
        df_updated = df.copy()
        for col in  df.columns:
            if '_x' in col.lower():
                df_updated[col] = df['BMR_Middle_x'] -df[col]
            elif '_y' in col.lower():
                df_updated[col] = df['BMR_Middle_y'] -df[col]
        return df_updated
    '''
    Remove the columns related with the middle
    '''
    def remove_columns_with_middle(self, df): 
        df_cleaned = df.loc[:,~df.columns.str.lower().str.contains("middle")]
        return df_cleaned  
     
    '''
    interpolate nan data for each columns
    '''   
    def interpolate_nan_data(self, df):
        df_interpolate = df.copy()
        for col in df.columns:  
            df_interpolate[col] = df[col].interpolate().fillna(method="bfill").fillna(method="ffill").values  
        return df_interpolate    
    
    '''
    save as csv
    '''
    def save_as_csv(self, updated_df):
        file_name = os.path.basename(self._filename)
        name_without_ext = os.path.splitext(file_name)[0]
        directory = os.path.dirname(self._filename)
        output_file = os.path.join(directory, name_without_ext + ".csv")
        #convert to number to avoid errors in cvs
       
        updated_df.to_csv(output_file, index=False,quoting=0)
        
        
    
        
    def __call__(self):
        df = self.read_excel()
        if df is not None:
           updated_df = self.remove_data_outside_tube(df)
           updated_df = self.fill_no_detection(updated_df)
           updated_df = self.remove_middle(updated_df)
           updated_df = self.remove_columns_with_middle(updated_df)
           updated_df = self.interpolate_nan_data(updated_df)
           self.save_as_csv(updated_df)
           print("The csv file was saved")
        else:
            updated_df = None
       
                
        
        