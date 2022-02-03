import cv2
import numpy as np

#taskbar setup for dynamic change of taskbar values
def nothing(x):
    #function to pass to trackbar
    pass

def init_trackbars():
    cv2.namedWindow("Test Trackbars")
    cv2.createTrackbar("L-H", "Test Trackbars", 164, 255, nothing)
    cv2.createTrackbar("L-S", "Test Trackbars", 46, 255, nothing)
    cv2.createTrackbar("L-V", "Test Trackbars", 97, 180, nothing)
    cv2.createTrackbar("U-H", "Test Trackbars", 186, 255, nothing)
    cv2.createTrackbar("U-S", "Test Trackbars", 159, 255, nothing)
    cv2.createTrackbar("U-V", "Test Trackbars", 154, 180, nothing)

def trackbars(hsv_frame):
    l_h = cv2.getTrackbarPos("L-H", "Test Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Test Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Test Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Test Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Test Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Test Trackbars")

    lower_limit = np.array([l_h, l_s, l_v])
    upper_limit = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)

    return mask

def ret_masks(hsv_frame):
    #red mask#
    lower_limit = np.array([164, 46, 97])
    upper_limit = np.array([186, 159, 154])
    red_mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)
    #green mask#
    lower_limit = np.array([86, 71, 79])
    upper_limit = np.array([104, 119, 122])
    green_mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)
    #blue mask#
    lower_limit = np.array([93, 85, 98])
    upper_limit = np.array([120, 172, 157])
    blue_mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)
    #yellow mask - needs more adjustment#
    lower_limit = np.array([15, 54, 110])
    upper_limit = np.array([28, 123, 168])
    yellow_mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)

    return red_mask, green_mask, blue_mask, yellow_mask

def draw_contours(frame ,red_mask, green_mask, blue_mask, yellow_mask):
    _, contours_red     , _    = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, contours_green   , _    = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, contours_blue    , _    = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, contours_yellow  , _    = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        if area > 400:
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)

            if len(approx) == 3:
                print("Crveni Trougao")
                writeCntImg(frame, cnt)
            elif len(approx) > 7 and len(approx) < 15:
                print("Crveni Krug")
                writeCntImg(frame, cnt)

    for cnt in contours_green:
        area = cv2.contourArea(cnt)
        if area > 400:
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 3)

            if len(approx) == 4:
                print("Zeleni Pravougaonik")
                writeCntImg(frame, cnt)

    for cnt in contours_blue:
        area = cv2.contourArea(cnt)
        if area > 400:
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            cv2.drawContours(frame, [cnt], 0, (255, 0, 0), 3)

            if len(approx) == 4:
                print("Plavi Pravougaonik")
                writeCntImg(frame, cnt)

    for cnt in contours_yellow:
        area = cv2.contourArea(cnt)
        if area > 400:
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            cv2.drawContours(frame, [cnt], 0, (0, 164, 186), 3)

            if len(approx) == 4:
                print("Zuti Pravougaonik")
                writeCntImg(frame, cnt)    

def filtering(masks):
    for mask in masks:
        kernel1 = np.ones((5,5), np.uint8)
        kernel2 = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel1)
        #mask = cv2.dilate(mask, kernel2)

def writeCntImg(frame ,cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    output = frame[y:y+h, x:x+w]
    cv2.imwrite('output.jpg', output)
    
#Use if you need to find hsv parameters. 
#Note that you should comment out the cv2.imshow("img", frame) in main while loop
def testMask(frame, hsv_frame):
    test_mask = trackbars(hsv_frame)
    filtering([test_mask])
    _, contours, _ = cv2.findContours(test_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)

            if len(approx) == 3:
                print("Trougao")
            elif len(approx) == 4:
                print("Pravougaonik")
            elif len(approx) > 7 and len(approx) < 15:
                print("Krug")
            else:
                print("Nije")

        
     
    cv2.imshow("img", frame)
    cv2.imshow("test mask", test_mask)

cap = cv2.VideoCapture(0)
init_trackbars()
while True:
    ret, frame = cap.read()

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #testMask(frame, hsv_frame)
 
    red_mask, green_mask, blue_mask, yellow_mask = ret_masks(hsv_frame)
    filtering([red_mask, green_mask, blue_mask, yellow_mask])
    draw_contours(frame ,red_mask, green_mask, blue_mask, yellow_mask)
    cv2.imshow("img", frame)

    if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

 
#TO DO:
    #Ispeglati koeficijente za maske za svaku b#
    #Za zelenu boju koristiti poseban filtar tako da se crvena / linija izgubi na hsvu
    #Za sve boje osim crvene, posto ocekujemo samo pravougaonike, parametar epsilon iz funkcije approxPolyDP()..., 0.2*...) povec#
    #optimizuj i sredi #
    #dodati poseban slucaj za zutu boju, nije dovoljno traziti samo zutu na slici jer je znak prednosti bangav
    #DONE
