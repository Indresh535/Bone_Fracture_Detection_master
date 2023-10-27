from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from pre_process import _reshape_img, get_model, resizeImage
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

#==============Manual edge ditect=====================
def segment_img(_img,limit):
	for i in range(0,_img.shape[0]-1):
		for j in range(0,_img.shape[1]-1): 
			if int(_img[i,j+1])-int(_img[i,j])>=limit:
				_img[i,j]=0
			elif(int(_img[i,j-1])-int(_img[i,j])>=limit):
				_img[i,j]=0
	
	return _img
#======================================================

def detect_fracture(image_path):
    img_t = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    shape = img.shape       

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    median = cv2.medianBlur(gray, 5)

    model_name = "ridge_model"
    model = get_model(model_name)
    pred_thresh = model.predict([_reshape_img(img_t)])
    bool, threshold_img = cv2.threshold(median, int(pred_thresh), 255, cv2.THRESH_BINARY)

    initial = []
    final = []
    line = []

    for i in range(0, gray.shape[0]):
        tmp_initial = []
        tmp_final = []
        for j in range(0, gray.shape[1] - 1):
            if threshold_img[i, j] == 0 and (threshold_img[i, j + 1]) == 255:
                tmp_initial.append((i, j))
            if threshold_img[i, j] == 255 and (threshold_img[i, j + 1]) == 0:
                tmp_final.append((i, j))
        x = [each for each in zip(tmp_initial, tmp_final)]
        x.sort(key=lambda each: each[1][1] - each[0][1])
        try:
            line.append(x[len(x) - 1])
        except IndexError:
            pass

    err = 15
    danger_points = []
    dist_list = []

    for i in range(1, len(line) - 1):
        dist_list.append(line[i][1][1] - line[i][0][1])
        try:
            prev_ = line[i - 3]
            next_ = line[i + 3]

            dist_prev = prev_[1][1] - prev_[0][1]
            dist_next = next_[1][1] - next_[0][1]
            diff = abs(dist_next - dist_prev)
            if diff > err:
                data = (diff, line[i])
                if len(danger_points):
                    prev_data = danger_points[len(danger_points) - 1]
                    if abs(prev_data[0] - data[0]) > 2 or data[1][0] - prev_data[1][0] != 1:
                        danger_points.append(data)
                else:
                    danger_points.append(data)
        except Exception as e:
            pass

        if len(danger_points) >= 2:
            for i in range(0, len(danger_points) - 1, 2):
                try:
                    start_rect = danger_points[i][1][0][::-1]
                    start_rect = (start_rect[0] - 40, start_rect[1] - 40)

                    end_rect = danger_points[i + 1][1][1][::-1]
                    end_rect = (end_rect[0] + 40, end_rect[1] + 40)

                    cv2.rectangle(img, start_rect, end_rect, (0, 255, 0), 2)
                except:
                    pass
        else:
            try:
                start_rect = danger_points[0][1][0][::-1]
                start_rect = (start_rect[0] - 40, start_rect[1] - 40)

                end_rect = danger_points[0][1][1][::-1]
                end_rect = (end_rect[0] + 40, end_rect[1] + 40)

                cv2.rectangle(img, start_rect, end_rect, (0, 255, 0), 2)
            except:
                pass


    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig2, ax3 = plt.subplots(1, 1)

    x = np.arange(1, gray.shape[0] - 1)
    y = dist_list

    cv2.calcHist(gray, [0], None, [256], [0, 256])

    try:
        ax1.plot(x, y)
    except:
        pass

    #img = np.rot90(img)
    ax2.imshow(img)

    ax3.hist(gray.ravel(), 256, [0, 256])

    # plt.show()

    prediction_result = "Fractured" if len(danger_points) >=1 else "Not Fractured"

    return prediction_result, danger_points, img



@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Handle uploaded image
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                input_image_path = 'uploads/input_image.jpg'
                output_image_path = 'static/Predicted_Output_Images/Predicted_Output_Images.jpg'
                file.save(input_image_path)
                
                allowed_extensions = {'png'}                
                extension = file.filename.split(".")[-1].lower()
                file_size = os.path.getsize(input_image_path)
                if file_size > 25 * 1024:  # Check if file size is greater than 25 KB                    
                    outimg = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
                    cv2.imwrite(output_image_path, outimg)                    
                    return render_template('predict.html', prediction_result="Invalid Image",danger_points="None", predicted_img=output_image_path)                
                                
                if extension not in allowed_extensions:  # Check if image extension is equal to .png
                    outimg = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
                    cv2.imwrite(output_image_path, outimg)
                    return render_template('predict.html', prediction_result="Invalid Image",danger_points="None", predicted_img=output_image_path)                
                                
                resizeImage(input_image_path)
                reszied_image_path = 'images/resized/resized.jpg'
                prediction_result, danger_points, img = detect_fracture(reszied_image_path)
                cv2.imwrite('static/Predicted_Output_Images/Predicted_Output_Images.jpg', img)  # Save the processed image
                
                return render_template('predict.html', prediction_result=prediction_result, danger_points=danger_points, predicted_img=img)
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
