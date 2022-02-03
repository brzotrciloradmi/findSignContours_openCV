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
    lower_limit = np.array([160, 66, 120])
    upper_limit = np.array([190, 178, 146])
    red_mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)
    #green mask#
    lower_limit = np.array([83, 61, 83])
    upper_limit = np.array([105, 107, 127])
    green_mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)
    #blue mask#
    lower_limit = np.array([104, 89, 95])
    upper_limit = np.array([109, 143, 133])
    blue_mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)
    #yellow mask - needs more adjustment#
    lower_limit = np.array([0, 31, 134])
    upper_limit = np.array([42, 98, 158])
    yellow_mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)

    return red_mask, green_mask, blue_mask, yellow_mask

def draw_contours(frame ,red_mask, green_mask, blue_mask, yellow_mask):
    _, contours_red     , _    = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, contours_green   , _    = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, contours_blue    , _    = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, contours_yellow  , _    = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_red + contours_green + contours_blue #dodati yellow
    for cnt in contours:
        cv2.drawContours(frame, [cnt], 0, (0, 128, 0), 3)

def filtering(mask):
    kernel1 = np.ones((5,5), np.uint8)
    kernel2 = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel1)
    #mask = cv2.dilate(mask, kernel2)
#Use if you need to find hsv parameters. 
#Note that you should comment out the cv2.imshow("img", frame) in main while loop
def testMask(frame, hsv_frame):
    test_mask = trackbars(hsv_frame)
    filtering(test_mask)
    _, contours, _ = cv2.findContours(test_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            cv2.drawContours(frame, [approx], 0, (0, 128, 0), 3)

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

    testMask(frame, hsv_frame)
 
    #red_mask, green_mask, blue_mask, yellow_mask = ret_masks(hsv_frame)
    #draw_contours(frame ,red_mask, green_mask, blue_mask, yellow_mask)

    #cv2.imshow("img", frame)
    #cv2.imshow("Red mask", red_mask)
    #cv2.imshow("Green mask", green_mask)

    if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

 
#TO DO:
    #1. Ispeglati koeficijente za maske za svaku boju

    #2. Prosiriti kod da radi za sve boje(prosiriti draw_contours)

    #3. Za sve boje osim crvene, posto ocekujemo samo pravougaonike, parametar epsilon iz funkcije approxPolyDP()..., 0.2*...) povecati

    #4. optimizuj i sredi kod

    #5. na osnovu aproksimovanih kontura izvuci najmanju i najvecu x i y koordinatu i parsirati taj isecak u novu sliku

    #DONE
