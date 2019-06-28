import cv2
import numpy as np
import getplate
import pytesseract
import os
import time

const_sub = 1.1
const_subh = 1.07
Loop = True
Específic = "br/AZJ6991.jpg"
Mostrar_caracteres = False
Mostrar_original = False
Debug_v = False
Debug_bin_v = False
benchmark = False
benchmark1=0
benchmark2=0
benchmark3=0
benchmark4=0
benchmark5=0
benchmark6=0
benchmark7=0
benchmark8=0
debugbenchmark = False
subdebug = False
debugnajuste = False

tam_kernel2 = 2
gaussian_blur = 3
tam_kernel = 15

def binarization(imgg,delta,n):
	imgg=cv2.adaptiveThreshold(imgg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,n,delta)
	imgg = cv2.GaussianBlur(imgg,(gaussian_blur,gaussian_blur),0)
	a,imgg = cv2.threshold(imgg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	kernel = np.ones((tam_kernel2,tam_kernel2), np.uint8)
	imgg = cv2.morphologyEx(imgg, cv2.MORPH_CLOSE, kernel)
	return imgg

def Teste(org, org1,ad):
	global benchmark1
	global benchmark2
	global benchmark3
	global benchmark4
	global benchmark5
	global benchmark6
	global benchmark7
	global benchmark8
	org5 = org1.copy()
	org6 = org1.copy()
	img = cv2.cvtColor(org1, cv2.COLOR_BGR2GRAY)
	org1 = img.copy()
	img = abs(255-img)
	kernel = np.ones((tam_kernel,tam_kernel), np.uint8)
	img1 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
	img2 = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
	img = cv2.add(img,img1)
	img = cv2.absdiff(img,img2)
	treshh = img.copy()
	treshh2 = abs(255-treshh)
	copy = img.copy()
	ii=0
	i=0
	delta = -5
	n = 401
	yy=[]
	hh=[]
	ww=[]
	while(i<7):
		yy.clear()
		hh.clear()
		ww.clear()
		i=0
		if(Debug_v):
			print("Tentando com: delta = "+str(delta)+" | n = "+str(n))
		ii=ii+1
		img = binarization(copy,delta,n)
		contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		for cnt in contours:
			x,y,w,h = cv2.boundingRect(cnt)
			if((w>8)&(w<70)&(h>100)&(h<160)&(y>2)&(x>6)):
				c=pytesseract.image_to_string(treshh2[y-2:y+h+4, x-6:x+w+4],config='--psm 10 -l Mandatory -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --oem 0')
				if((c=='A')|(c=='B')|(c=='C')|(c=='D')|(c=='E')|(c=='F')|(c=='G')|(c=='H')|(c=='I')|(c=='J')|(c=='K')|(c=='L')|(c=='M')|(c=='N')|(c=='O')|(c=='P')|(c=='Q')|(c=='R')|(c=='S')|(c=='T')|(c=='U')|(c=='V')|(c=='W')|(c=='X')|(c=='Y')|(c=='Z')|(c=='0')|(c=='1')|(c=='2')|(c=='3')|(c=='4')|(c=='5')|(c=='6')|(c=='7')|(c=='8')|(c=='9')):
					yy.append(y)
					hh.append(h)
					ww.append(w)
					if(Debug_v):
						print(c+" | "+str(i))
					i=i+1
		if(ii==1):
			n = 101
		elif(ii==2):
			n = 201
		elif(ii==3):
			n = 801
		elif(ii==4):
			n = 2001
		elif(ii==5):
			n = 51
		elif(ii==6):
			delta = 0
			n = 401
		elif(ii==7):
			n = 101
		elif(ii==8):
			n = 201
		elif(ii==9):
			n = 801
		elif(ii==10):
			n = 2001
		elif(ii==11):
			n = 51
		elif(ii==12):
			delta = -20
			n = 401
		elif(ii==13):
			n = 101
		elif(ii==14):
			n = 201
		elif(ii==15):
			n = 801
		elif(ii==16):
			n = 2001
		elif(ii==17):
			n = 51
		elif(ii==18):
			delta = -30
			n = 401
		elif(ii==19):
			n = 101
		elif(ii==20):
			n = 201
		elif(ii==21):
			n = 801
		elif(ii==22):
			n = 2001
		elif(ii==23):
			n = 51
		elif(ii==24):
			delta = +10
			n = 401
		elif(ii==25):
			n = 101
		elif(ii==26):
			n = 201
		elif(ii==27):
			n = 801
		elif(ii==28):
			n = 2001
		elif(ii==29):
			n = 51
		elif(ii==30):
			delta = +20
			n = 401
		elif(ii==31):
			n = 101
		elif(ii==32):
			n = 201
		elif(ii==33):
			n = 801
		elif(ii==34):
			n = 2001
		elif(ii==35):
			n = 51
		elif(ii==36):
			delta = -40
			n = 401
		elif(ii==37):
			n = 101
		elif(ii==38):
			n = 201
		elif(ii==39):
			n = 801
		elif(ii==40):
			n = 2001
		elif(ii==41):
			n = 51
		else:
			if(debugnajuste):
				print('Nenhum ajuste encontrado para gradiente invertida!')
			break
		if(Debug_bin_v):
			cv2.imshow('degub binarization',img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()	
	med_y=int(sum(yy)/7)
	med_h=int(sum(hh)/7)
	med_w=int(sum(ww)/7)
	copy = img.copy()
	crop_imgs = []
	crop_imgs2 = []
	crop_imgs3 = []
	crop_imgs4 = []
	contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		if((w>8)&(w<70)&(h>100)&(h<160)&(y>3)&(x>4)):
			if(w>(med_w*const_sub)):
				x=int(x+(w-med_w)/2)
				w=med_w
				if(subdebug):
					print('subw')
			if((h>(med_h*const_subh))|(h<(med_h/const_subh))):
				h=med_h
				if(subdebug):
					print('subh')
			crop_imgs.append((treshh[y-3:y+h+6, x-4:x+w+6],x))
			crop_imgs2.append((copy[y-3:y+h+6, x-4:x+w+6],x))
			crop_imgs3.append((org1[y-3:y+h+6, x-4:x+w+6],x))
			crop_imgs4.append((treshh2[y-3:y+h+6, x-4:x+w+6],x))
			org6 = cv2.rectangle(org6,(x-4,y-3),(x+w+6,y+h+6),(255,255,0),1)
		org5 = cv2.rectangle(org5,(x,y),(x+w,y+h),(255,255,0),1)

	crop_imgs.sort(key = lambda x: int(x[1]))
	crop_imgs2.sort(key = lambda x: int(x[1]))
	crop_imgs3.sort(key = lambda x: int(x[1]))
	crop_imgs4.sort(key = lambda x: int(x[1]))
	Placa=''
	Placa2=''
	Placa3=''
	Placa4=''
	if(Mostrar_original):
		cv2.imshow('Completa',org)

	for i in range(len(crop_imgs)):
		if(Mostrar_caracteres):
			cv2.imshow(('Gradiente '+str(i)),crop_imgs[i][0])

		if(i<3):
			Placa = Placa + pytesseract.image_to_string(crop_imgs[i][0],config='--psm 10 -l Mandatory -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --oem 0')
		else:
			Placa = Placa + pytesseract.image_to_string(crop_imgs[i][0],config='--psm 10 -l Mandatory -c tessedit_char_whitelist=0123456789 --oem 0')

	for i in range(len(crop_imgs2)):
		if(Mostrar_caracteres):
			cv2.imshow(('Binarizada '+str(i)),crop_imgs2[i][0])

		if(i<3):
			Placa2 = Placa2 + pytesseract.image_to_string(crop_imgs2[i][0],config='--psm 10 -l Mandatory -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --oem 0')
		else:
			Placa2 = Placa2 + pytesseract.image_to_string(crop_imgs2[i][0],config='--psm 10 -l Mandatory -c tessedit_char_whitelist=0123456789 --oem 0')

	for i in range(len(crop_imgs3)):
		if(Mostrar_caracteres):
			cv2.imshow(('Original '+str(i)),crop_imgs3[i][0])

		if(i<3):
			Placa3 = Placa3 + pytesseract.image_to_string(crop_imgs3[i][0],config='--psm 10 -l Mandatory -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --oem 0')
		elif(i<7):
			Placa3 = Placa3 + pytesseract.image_to_string(crop_imgs3[i][0],config='--psm 10 -l Mandatory -c tessedit_char_whitelist=0123456789 --oem 0')

	for i in range(len(crop_imgs4)):
		if(Mostrar_caracteres):
			cv2.imshow(('Gradiente invertido '+str(i)),crop_imgs4[i][0])

		if(i<3):
			Placa4 = Placa4 + pytesseract.image_to_string(crop_imgs4[i][0],config='--psm 10 -l Mandatory -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --oem 0')
		elif(i<7):
			Placa4 = Placa4 + pytesseract.image_to_string(crop_imgs4[i][0],config='--psm 10 -l Mandatory -c tessedit_char_whitelist=0123456789 --oem 0')
	

	Placa5 = pytesseract.image_to_string(treshh,config='--psm 9 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
	Placa6 = pytesseract.image_to_string(copy,config='--psm 9 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
	Placa7 = pytesseract.image_to_string(org1,config='--psm 9 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
	Placa8 = pytesseract.image_to_string(treshh2,config='--psm 9 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
	if((benchmark==False)|debugbenchmark):
		print("Placa lida dos characteres da imagem gradiente: "+Placa)
		print("Placa lida dos characteres da imagem binarizada: "+Placa2)
		print("Placa lida dos characteres da imagem original: "+Placa3)
		print("Placa lida dos characteres da imagem gradiente invertida: "+Placa4)
		print("Placa lida da imagem gradiente: "+Placa5)
		print("Placa lida da imagem binarizada: "+Placa6)
		print("Placa lida da imagem original: "+Placa7)
		print("Placa lida da imagem gradiente invertida: "+Placa8)

	if((benchmark==False)|debugbenchmark):
		cv2.imshow('Gradiente',treshh)
		cv2.imshow('Binarizada',copy)
		cv2.imshow('Original',org1)
		cv2.imshow('Gradiente invertida',treshh2)
		cv2.imshow('Todos os contornos',org5)
		cv2.imshow('Contornos Identificados como Caracter',org6)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	if(Placa==str(ad[0:7])):
		benchmark1=benchmark1+1
	if(Placa2==str(ad[0:7])):
		benchmark2=benchmark2+1
	if(Placa3==str(ad[0:7])):
		benchmark3=benchmark3+1
	if(Placa4==str(ad[0:7])):
		benchmark4=benchmark4+1
	if(Placa5==str(ad[0:7])):
		benchmark5=benchmark5+1
	if(Placa6==str(ad[0:7])):
		benchmark6=benchmark6+1
	if(Placa7==str(ad[0:7])):
		benchmark7=benchmark7+1
	if(Placa8==str(ad[0:7])):
		benchmark8=benchmark8+1



benchmarkcount=len(os.listdir('br/'))
if(Loop):
	for ad in os.listdir('br/'):
		print('Lendo: br/'+ad)
		org = cv2.imread("br/"+ad)
		org1 = getplate.getplate("br/"+ad)
		org1 = cv2.resize(org1,(512,200))
		Teste(org, org1,ad)
elif(benchmark):
	inicio = time.time()
	print('Iniciando Benchmark!')
	i=0;
	for ad in os.listdir('br/'):
		i=i+1
		print('Iniciando Loop para: '+ad+' | Loop '+str(i)+'/'+str(benchmarkcount))
		org = cv2.imread("br/"+ad)
		org1 = getplate.getplate("br/"+ad)
		org1 = cv2.resize(org1,(512,200))
		Teste(org, org1,ad)
		if(debugbenchmark):
			print('Primeiro Loop: '+str(benchmark1)+' | '+str(benchmark2)+' | '+str(benchmark3)+' | '+str(benchmark4))
	print('Porcentagem de acerto da placa lida dos characteres da imagem Gradiente: '+ str(100*(benchmark1/benchmarkcount)))
	print('Porcentagem de acerto da placa lida dos characteres da imagem Binarizada: '+ str(100*(benchmark2/benchmarkcount)))
	print('Porcentagem de acerto da placa lida dos characteres da imagem Original: ' +str(100*(benchmark3/benchmarkcount)))
	print('Porcentagem de acerto da placa lida dos characteres da imagem Gradiente invertido: '+ str(100*(benchmark4/benchmarkcount)))
	print('Porcentagem de acerto da placa lida da imagem Gradiente: '+ str(100*(benchmark5/benchmarkcount)))
	print('Porcentagem de acerto da placa lida da imagem Binarizada: '+ str(100*(benchmark6/benchmarkcount)))
	print('Porcentagem de acerto da placa lida imagem Original: ' +str(100*(benchmark7/benchmarkcount)))
	print('Porcentagem de acerto da placa lida imagem Gradiente invertido: '+ str(100*(benchmark8/benchmarkcount)))
	fim = time.time()
	print('Tempo Decorrido:')
	print(fim-inicio)
	
else:
	org = cv2.imread(Específic)
	org1 = getplate.getplate(Específic)
	org1 = cv2.resize(org1,(512,200))
	Teste(org, org1,Específic)


