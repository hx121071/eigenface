import cv2
import skimage.io as si
import numpy as np

window =50
def PCA(x):
    '''x.shape = *32*32'''
    x_reshape = x.reshape(x.shape[0],-1)
    x_mean= np.mean(x_reshape,axis=0)
# face_candidate_r=reconstruct(face_candidate-x_mean,x_mean,V_k)
    # im_blob_r=reconstruct(im_blob.reshape(21,-1)-x_mean,x_mean,V_k)
    X=x_reshape-x_mean
    conv_x=X.T.dot(X)
    [U,S,V] = np.linalg.svd(conv_x)
    sum1=np.sum(S)
    temp_sum=0

    for  i in range(len(S)):
        temp_sum=np.sum(S[:i+1])
        if  temp_sum/sum1>0.99 :
            print(i)
            V_k=V[:i+1,:].T
            break
    return X,x_mean,V_k

def reconstruct(x,x_mean,V_k):
    x_hat=x.dot(V_k).dot(V_k.T)
    x_result=x_hat+x_mean
    return x_result

def projection(x,V_k):
    return x.dot(V_k)

def get_min(sub_window):
    win=sub_window.reshape(-1)
    return np.min(win)
def face_detection(im_test,x_mean,V_k,im_blob,im_blob_feature,candidate_num=10):
    n,m=im_test.shape
    im= cv2.GaussianBlur(im_test,(9,9),0)
    scores=np.zeros(im.shape)
    face_candidate=np.zeros([candidate_num,window*window])
    face_candidate_gt=np.zeros([candidate_num,window*window])
    face_rectangle=np.zeros([candidate_num,4])
    scores[np.where(scores==0)]=10000
    for i in range(0,n-window,4):
        for j in range(0,m-window,4):
            im_grid=im[i:i+window,j:j+window].copy()
            # method 1:
            im_blob.reshape(21,-1)# im_grid_r=im_grid.reshape(-1)-x_mean
            # im_grid_rr=im_grid_r.dot(V_k)
            # temp_scores=np.linalg.norm(im_grid_rr)
            #im_grid_rr=reconstruct(im_grid.reshape(-1),x_mean,V_k)

            # method 2:reconstructed face differences with the current subwindow
            im_grid_r=im_grid.reshape(-1)-x_mean
            im_grid_rr=reconstruct(im_grid.reshape(-1),x_mean,V_k)
            temp_scores=np.linalg.norm(im_grid_r-im_grid_rr)


            # method 3:original face images differences with the current subwindow
            # im_blob=im_blob.reshape(im_blob.shape[0],-1)
            # temp_scores=np.min(np.linalg.norm(im_blob-im_grid.reshape(-1),axis=1))
            scores[i,j]=temp_scores
    sub =int(window)
    for i in range(sub,n-sub,int(sub/2)):
        for j in range(sub,m-sub,int(sub/2)):
            min_s=get_min(scores[i-sub:i+sub,j-sub:j+sub])
            scores[np.where(scores[i-sub:i+sub,j-sub:j+sub] > min_s)[0]+i-sub,\
            np.where(scores[i-sub:i+sub,j-sub:j+sub] > min_s)[1]+j-sub]=10000

    temp_scores = scores.reshape(-1)
    # print(temp_scores.shape)
    index=np.argsort(temp_scores)


    x_i=np.zeros((candidate_num,),dtype=np.int)
    for i in range(candidate_num):
        x_i[i]=int(index[i]/m)

    x_j=index[0:candidate_num]%m
    current_face_candidate=0
    for i in range(candidate_num):
        if temp_scores[index[i]] != 10000:
            im_grid=im[x_i[i]:x_i[i]+window,x_j[i]:x_j[i]+window].reshape(-1)
            im_grid_gt=im_test[x_i[i]:x_i[i]+window,x_j[i]:x_j[i]+window].reshape(-1)
            face_candidate[current_face_candidate,:]=im_grid
            face_candidate_gt[current_face_candidate,:]=im_grid_gt
            face_rectangle[i,:]=np.array([x_j[i]+window,x_i[i]+window,x_j[i],x_i[i]],dtype=np.int)
            # si.imsave("output/fd0{:d}.tga".format(i+1),im[x_i[i]:x_i[i]+window,x_j[i]:x_j[i]+window])
            # cv2.rectangle(im,(x_j[i]+window,x_i[i]+window),(x_j[i],x_i[i]),(1.0,1.0,1.0),1)
            current_face_candidate+=1
    # cv2.imshow("output",im)
    # cv2.waitKey(0)
    return face_candidate,face_candidate_gt,face_rectangle
def show_result(face_candidate,groud_truth_face,scores,face_rectangle,im):
    #fc=face_candidate.reshape(window,window)
    #gtf=groud_truth_face.reshape(window,window)
    #arreagte=np.zeros([window,2*window])
    #arreagte[:,0:window]=fc
    #arreagte[:,window:2*window]=gtf
    cv2.rectangle(im,(int(face_rectangle[0]),int(face_rectangle[1])),(int(face_rectangle[2]),int(face_rectangle[3])),(1.0,1.0,1.0),1)
    cv2.putText(im,scores,(int(face_rectangle[0]),int(face_rectangle[1])), cv2.FONT_HERSHEY_SIMPLEX,0.5,(1.0,1.0,1.0),1,1)



def face_recognize(im,face_candidate_gt,face_candidate,face_rectangle,groud_truth_face,im_blob,x_mean,V_k,im_i):
    #method 1:
    face_candidate_r=reconstruct(face_candidate-x_mean,x_mean,V_k)
    im_blob_r=reconstruct(im_blob.reshape(21,-1)-x_mean,x_mean,V_k)
    #method 2:
    # face_candidate_r=face_candidate-x_mean
    # im_blob_r=im_blob.reshape(21,-1)-x_mean
    #method 3:
    # face_candidate_r=(face_candidate-x_mean).dot(V_k)
    # im_blob_r=(im_blob.reshape(21,-1)-x_mean).dot(V_k)
    num_face_can=face_candidate.shape[0]
    
    for i in range(num_face_can):
        scores=np.linalg.norm(im_blob_r-face_candidate_r[i,:],axis=1)
        index=np.argsort(scores)[0]
        print(scores[index])
        show_result(face_candidate_gt[i,:],groud_truth_face[index],str(index+1),face_rectangle[i,:],im)
    cv2.imshow("examples",im)
    si.imsave("output/fd0{:d}.tga".format(im_i+1),im)
    cv2.waitKey(0)


def read_save_im(filename):
    f=open(filename,"r")

    im_list=[]
    for line in f.readlines():
        line_split=line.split('\n')
        im_temp=si.imread(line_split[0],True)
        im_list.append(im_temp)
    return im_list


im_list=read_save_im("smiling_cropped/list.txt")
im_resize_list=[]

im_scale_factors=[]
for im in im_list:
    im_blur=cv2.GaussianBlur(im,(5,5),0)
    im_resize=cv2.resize(im_blur,(window,window),interpolation=cv2.INTER_LINEAR)
    im_resize_list.append(im_resize)


im_blob = np.zeros([len(im_list),window,window])



for i in range(len(im_resize_list)):
    #print(np.shape(im_resize_list[i]))
    im_blob[i,:,:]=im_resize_list[i]

#PCA
X,x_mean,V_k=PCA(im_blob)
#reconstruct
x_result=reconstruct(X,x_mean,V_k)
#projection
im_blob_feature = (im_blob.reshape(21,-1)-x_mean).dot(V_k)

#si.imsave("./output/mean.tga",x_mean.reshape(window,window))

##image detections and recognize
f_test=open("group/smiling/list.txt","r")
candidate_num=5
face_candidate=np.zeros([candidate_num,window*window])
face_candidate_gt=np.zeros([candidate_num,window*window])
face_rectangle=np.zeros([candidate_num,4])
im_i=0
groud_truth=im_blob.reshape(im_blob.shape[0],-1)
for line in f_test.readlines():
    line_split=line.split('\n')
    im_test =si.imread(line_split[0],True)

    face_candidate,face_candidate_gt,face_rectangle=\
    face_detection(im_test,x_mean,V_k,im_blob,im_blob_feature,candidate_num)
    face_recognize(im_test,face_candidate_gt,face_candidate,face_rectangle,groud_truth,im_blob,x_mean,V_k, im_i)
    im_i+=1





# for i in range(10):
#     eigen_face=V_k[:,i].T.reshape(window,window)
#     si.imsave('output/e0{:d}.tga'.format(i+1),eigen_face)
# x=x_result.reshape(-1,window,window)
# for i in range(x.shape[0]):
#     si.imsave('output/0{:d}.tga'.format(i+1),x[i,:,:])
# for i in range(21):
#     cv2.imshow("test{:d}".format(i),x[i,:,:])
#     cv2.waitKey(100)
#
# cv2.waitKey(0)
# #
#
#
# print(x_hat)
# # cv2.imshow("test0",x_hat[0,:,:])
# # cv2.imshow("test1",x_hat[1,:,:])
# # cv2.waitKey(0)
