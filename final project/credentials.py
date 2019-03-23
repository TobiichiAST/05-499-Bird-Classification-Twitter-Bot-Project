import tweepy
from tweepy import API
import credentials
import json
from itertools import groupby
import wget
import PIL
from PIL import Image

import scipy
import numpy as np
import glob
import os
from skimage import io, color
import sklearn
from skimage import data
from skimage import filters
from skimage import exposure

from sklearn.naive_bayes import MultinomialNB
import pickle
import imageio
import glob
import cv2
import random

def resize_and_cut(name, result_name):
    im = Image.open(name)
    mywidth = 192
    myheight = 192
    imagewidth = (im.size[0])
    imageheight = (im.size[1])
    if (imagewidth < imageheight):
        wpercent = (mywidth/float(imagewidth))
        hsize = int((float(imageheight)*float(wpercent)))
        im = im.resize((mywidth,hsize), PIL.Image.ANTIALIAS)
        difference = (hsize - myheight) / 2
        area = (0, difference, mywidth, hsize - difference)
        cropped_img = im.crop(area)
        cropped_img.save(result_name)
    else:
        hpercent = (myheight/float(imageheight))
        wsize = int((float(imagewidth)*float(hpercent)))
        im = im.resize((wsize,myheight), PIL.Image.ANTIALIAS)
        difference = (wsize - mywidth) /2
        area = (difference, 0, wsize - difference, myheight)
        cropped_img = im.crop(area)
        cropped_img.save(result_name)

def change_image(name):
    windowName = "Edges"
    pictureRaw = cv2.imread(name)
    
    #the erosion + dilation
    pictureGray = cv2.cvtColor(pictureRaw,  cv2.COLOR_BGR2GRAY)
    pictureGaussian = cv2.GaussianBlur(pictureGray, (21,21), 0)
    pictureCanny = cv2.Canny(pictureGaussian, 50, 100)
    pictureDilate = cv2.dilate(pictureCanny, None, iterations= 50)
    pictureErode = cv2.erode(pictureDilate, None, iterations=5)

    #Otsu thresholding
    Otsu = pictureRaw
    Otsugray = cv2.cvtColor(Otsu,cv2.COLOR_BGR2GRAY)
    Otsuret, Otsuthresh = cv2.threshold(Otsugray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    #combined
    combined = Otsuthresh + pictureErode

    #masked result
    imask = combined > 0
    canvas = np.full_like(pictureRaw, np.array([0,0,0]), dtype=np.uint8)
    canvas[imask] = pictureRaw[imask]


    #grab cut
    grabimg = pictureRaw
    grabmask = np.zeros(grabimg.shape[:2],np.uint8)
    grabbgdModel = np.zeros((1,65),np.float64)
    grabfgdModel = np.zeros((1,65),np.float64)
    grabrect = (50,50,1000,1000)
    try:
        cv2.grabCut(grabimg,grabmask,grabrect,grabbgdModel,grabfgdModel,5,cv2.GC_INIT_WITH_RECT)
    except:
        print(name)
    grabmask2 = np.where((grabmask==2)|(grabmask==0),0,1).astype('uint8')
    grabimg = grabimg*grabmask2[:,:,np.newaxis]

    #combined of all things
    #first filter the image by applying erosions + dilations and Otsu
    #then filter the resulting image by applying grab Cut
    combinedImage = (canvas*grabmask2[:,:,np.newaxis])[:,:,::-1]
    imageio.imsave(name, combinedImage)
    
    

def open_model(name):
    f = open(name + '.pckl', 'rb')
    test = pickle.load(f)
    f.close()
    return test

def main():
	auth = tweepy.OAuthHandler("your stuff here")
	auth.set_access_token("your stuff here")
	api = tweepy.API(auth)

	user = api.me()


	###
	# Load the Response information
	###
	with open('replied.json') as json_file:  
	    replied = json.load(json_file)

	###
	# Load the models
	###
	model1 = open_model('model1')
	model2 = open_model('model2')


	####
	# Define the search
	#####
	query = '@MarkHe48478372'
	max_tweets = 100

	####
	# Do the search
	#####
	searched_tweets = []
	last_id = -1
	while len(searched_tweets) < max_tweets:
	    count = max_tweets - len(searched_tweets)
	    try:
	        new_tweets = api.search(q=query, count=count, max_id=str(last_id - 1))
	        if not new_tweets:
	            break
	        searched_tweets.extend(new_tweets)
	        last_id = new_tweets[-1].id
	    except tweepy.TweepError as e:
	        # depending on TweepError.code, one may want to retry or wait                                                                                                                 
	        # to keep things simple, we will give up on an error                                                                                                                          
	        break

	####
	# Iterate over the search
	#####

	for status in searched_tweets:
		reply_id = (status.id)
		user_name = (status.user.name)
	  # do something with all these tweets
		if (str(reply_id) not in replied):    
			print('not in')
			if ("media" in status._json['entities']):
				image_url = (status._json['entities']['media'][0]['media_url'])
				file_name = wget.download(image_url)
				resize_and_cut(file_name, 'model1.jpg')
				resize_and_cut(file_name, 'model2.jpg')
				change_image('model1.jpg')
				file_1 = cv2.imread('model1.jpg')
				file_2 = cv2.imread('model2.jpg')
				print()
				result_1 = ((model1.predict(np.array([((np.array((file_1))).flatten())])))[0])
				result_2 = ((model2.predict(np.array([((np.array((file_2))).flatten())])))[0])
				
				for folder_name in glob.glob('original_images/*'):
					if (folder_name[16:]).startswith(result_1):
						folder_1 = folder_name
					if (folder_name[16:]).startswith(result_2):
						folder_2 = folder_name
				imagerespond_1 = folder_1 + '/' + str(random.randint(1, len(glob.glob(folder_1 + '/*')))) + '.jpg'
				imagerespond_2 = folder_2 + '/' + str(random.randint(1, len(glob.glob(folder_2 + '/*')))) + '.jpg'
				# upload images and get media_ids
				filenames = [file_name, imagerespond_1, imagerespond_2]
				media_ids = []
				for filename in filenames:
				     res = api.media_upload(filename)
				     media_ids.append(res.media_id)
				# tweet with multiple images

				Response_1 = (folder_1[20:].replace("_", " "))
				Response_2 = (folder_2[20:].replace("_", " "))

				text = 'Hi ' + user_name + '! Your image(left) might be a ' + Response_1 + '(top right)🤔. Or it might be a '\
					+ Response_2 + "(bottom right)🤨. I'm not a super clever AI, so it might not be any of these. 🤪"\
					+ """\n(Don't take my words as "professional" result.)"""+ " @" + status.author.screen_name
				print(text)
				api.update_status(status=text, media_ids=media_ids)

				os.remove(file_name)
				os.remove('model1.jpg')
				os.remove('model2.jpg')
				replied[reply_id] = True

			else:
				replied[reply_id] = True
				api.update_status("I see no image of bird. ;) @" + status.author.screen_name, status.id_str)

	with open('replied.json', 'w') as fp:
		json.dump(replied, fp)

if (__name__== "__main__"):
	main()
