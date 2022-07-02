import pandas as pd
import numpy as np
import cv2 
import os
import requests
from PIL import Image
from io import BytesIO
import skimage
import os.path

class ImageProcessor():
    def __init__(self):
        pass

    def detect_colors(self,image_url,n_clusters=7, extract_overall_color=0):
        api_key = 'acc_e8757fe81b74a32'
        api_secret = 'c7c4aeb4a98bd40704fe023904297ad5'
        response = requests.get(
            'https://api.imagga.com/v2/colors?image_url=%s&separated_count=%s&extract_overall_colors=%s&overall_count=%s' % (image_url, n_clusters, extract_overall_color, n_clusters),
        auth=(api_key, api_secret))
        a = response.json()
        if(a["status"]["type"] == "success"):
            return(a)

    def load_img(self, url):
        response = requests.get(url)
        rgb_img = Image.open(BytesIO(response.content))
        rgb_img.thumbnail((300,300)) #resize pictures to selected maximum values for length and width
        return(rgb_img)
    
    def rgb2hsv(self, img): # reference : https://stackoverflow.com/questions/2659312/how-do-i-convert-a-numpy-array-to-and-display-an-image
        return(skimage.color.rgb2hsv(np.array(img)))
    
       
class ComplexityEstimator():
    def __init__(self):
        self.i_range, self.j_range = 9,9
        self.gama = 14
        
    def calculate_complexity_by_id(self, id):
        file_name = "ad_images/{}.png".format(id)
        if os.path.exists(file_name):
            img = cv2.imread(file_name)
            complexity = self.calculate_complexity(img)
            return complexity
        else:
            return(None)
        
    def calculate_complexity_fast(self, img):
        #link : https://towardsdatascience.com/one-simple-trick-for-speeding-up-your-python-code-with-numpy-1afc846db418
        
        arr = np.array(img)
        length = arr.shape[0]
        width = arr.shape[1]
        final_arr = []
        window_size = 9

        for i in range(window_size):
            for j in range(window_size):
                new_arr = arr[i:length-window_size+i, j:width-window_size+j,:]
                row = new_arr.reshape((length-window_size)*(width-window_size),3)

                final_arr.append(list(row))
        final_arr = np.array(final_arr)
        final_arr.shape
        c_bar = np.mean(final_arr, axis=0)
        subtracted = np.subtract(final_arr, c_bar)
        squared_subtracted = np.power(subtracted,2)
        euclidean_distance = np.sum(squared_subtracted, axis=2)

        sqrt = np.sqrt(euclidean_distance)
        diff = np.subtract(1,np.exp(-sqrt/14))

        mean_diff = np.mean(diff, axis=0)
        sigma_diff = np.var(diff, axis=0)

        p = np.divide(-np.power(np.subtract(diff,mean_diff),2), np.multiply(2,sigma_diff+0.0000001))
        gw = np.multiply(np.exp(p), diff)
        phi = np.sum(gw, axis= 0)
        complexity = np.mean(phi)

        return(complexity)
        
    def calculate_complexity(self, image): #image is an array with the shape of n*n*3
        # for i in windows:
        #     sum_of(calculate_window_phi)
        window_phi_list = []
        for i in range(image.shape[0]-9):
            for j in range(image.shape[1]-9):
                window = image[i:i+9, j:j+9, :]
                window_phi_list.append(self.calculate_window_phi(window))
        complexity = sum(window_phi_list)/((image.shape[0]-9)*(image.shape[1]-9))
        return(complexity)
        
        
        
    def calculate_window_phi(self, array): # input : 9*9*3 array of each pixel 
        color_differences = []
        window_mean_hue = np.mean(array[:,:,0])
        window_mean_saturation = np.mean(array[:,:,1])
        window_mean_value = np.mean(array[:,:,2])
        window_mean_color = [window_mean_hue, window_mean_saturation, window_mean_value]
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                color_differences.append(self.calculate_color_difference(array[i,j], window_mean_color))
        phi = self.calculate_gaussianweighted_sum(color_differences)
        return(phi)
        
    
    def calculate_color_difference(self, point_1, point_2):
        difference = 1 - np.exp(-(self.calculate_euclidean_distance(point_1,point_2)/self.gama))
        return(difference)
    
    def calculate_euclidean_distance(self, point_1, point_2):
        distance = np.sqrt((point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2 +(point_1[2]-point_2[2])**2)
        return(distance)
    
    def calculate_gaussianweighted_sum(self, array):
        mean = np.mean(array)
        sigma = np.var(array)
        if sigma == 0:
            return(0)
        weighted_array = []
        for a in array:
            weight = np.exp(-((a-mean)**2)/(2*(sigma)))
            weighted_array.append(weight*a)
        return(sum(weighted_array))

    def calculate_batch_complexities(self, ad_df):
        image_processor = ImageProcessor()
        for i, row in ad_df.iterrows():
            url = row.image_url
            
            try:
                img = image_processor.rgb2hsv(image_processor.load_img(url))
                ad_df.loc[i, "complexity"] = self.calculate_complexity_fast(img)
            except:
                print("error")
                ad_df.loc[i, "complexity"] = 0
        return(ad_df)
            


    def calculate_all_complexities(self, all_ad_df):
        all_ad_df["complexity"] = np.zeros(len(all_ad_df))
        if os.path.exists("labeled_ads.csv"):
            saved_df = pd.read_csv("labeled_ads.csv")
        else: 
            saved_df = pd.DataFrame([], columns=all_ad_df.columns)
        i = len(saved_df)
        while i+100 <len(all_ad_df):
            print(i)
            df = self.calculate_batch_complexities(all_ad_df.iloc[i:i+100, :])
            if os.path.exists("labeled_ads.csv"):
                saved_df = pd.read_csv("labeled_ads.csv", index_col=0)
            else: 
                saved_df = pd.DataFrame([], columns=all_ad_df.columns)
            final_df = saved_df.append(df)
            final_df.to_csv("labeled_ads.csv")


            i+=100

        if i < len(all_ad_df):
            df = self.calculate_batch_complexities(all_ad_df.iloc[i:, :])
            if os.path.exists("labeled_ads.csv"):
                saved_df = pd.read_csv("labeled_ads.csv", index_col=0)
            else: 
                saved_df = pd.DataFrame([], columns=all_ad_df.columns)
            final_df = saved_df.append(df)
            final_df.to_csv("labeled_ads.csv")



        

def main():
    ad_df = pd.read_csv("preprocessed_ad_ctr_df.csv", index_col=0)
    ce = ComplexityEstimator()
    ce.calculate_all_complexities(ad_df)

if __name__ == "__main__":
    main()

