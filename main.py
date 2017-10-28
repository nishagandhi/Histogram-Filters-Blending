# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def help_message():
    print("Usage: [Question_Number] [Input_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Histogram equalization")
    print("2 Frequency domain filtering")
    print("3 Laplacian pyramid blending")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to input image] " +
          "[output directory]")  # Single input, single output
    print(sys.argv[0] + " 2 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, three outputs
    print(sys.argv[0] + " 3 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, single output


# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================


def histogram_equalization(img_in):
    # Write histogram equalization here

    # Histogram equalization result

    blue, green, red = cv2.split(img_in)  #split the image into r,g,b channels
    hist_blue = cv2.calcHist(
        [blue], [0], None, [256],
        [0, 256])  #calculating the histogram and CDF for each histogram
    cdf_blue = np.cumsum(hist_blue)

    hist_green = cv2.calcHist([green], [0], None, [256], [0, 256])
    cdf_green = np.cumsum(hist_green)

    hist_red = cv2.calcHist([red], [0], None, [256], [0, 256])
    cdf_red = np.cumsum(hist_red)

    blue1 = np.around(np.subtract(cdf_blue, np.amin(cdf_blue)))
    cv2.divide(blue1, blue.size, blue1)
    cv2.multiply(blue1, 255, blue1)

    green1 = np.around(np.subtract(cdf_green, np.amin(cdf_green)))
    cv2.divide(green1, green.size, green1)
    cv2.multiply(green1, 255, green1)

    red1 = np.around(np.subtract(cdf_red, np.amin(cdf_red)))
    cv2.divide(red1, red.size, red1)
    cv2.multiply(red1, 255, red1)

    new_blue = blue1[blue.ravel()].reshape(blue.shape)
    new_green = green1[green.ravel()].reshape(green.shape)
    new_red = red1[red.ravel()].reshape(red.shape)

    img = cv2.merge([new_blue, new_green, new_red])  #Merging all channels
    img_out = img
    return True, img_out


def Question1():
    # Read in input images
    input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    # Histogram equalization
    succeed, output_image = histogram_equalization(input_image)

    # Write out the result
    output_name = sys.argv[3] + "1.jpg"
    cv2.imwrite(output_name, output_image)

    return True


# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================


def low_pass_filter(img_in):
    # Write low pass filter here
    dft = cv2.dft(np.float32(img_in), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(
        cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    rows, cols = img_in.shape
    crow, ccol = rows / 2, cols / 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 10:crow + 10, ccol - 10:ccol + 10] = 1
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    img_out = img_back

    return True, img_out  # Low pass filter result


def high_pass_filter(img_in):
    # Write high pass filter here
    dft = cv2.dft(np.float32(img_in), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(
        cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    rows, cols = img_in.shape
    crow, ccol = rows / 2, cols / 2
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow - 10:crow + 10, ccol - 10:ccol + 10] = 0
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    img_out = img_back  # High pass filter result

    return True, img_out


def deconvolution(img_in):
    # Write deconvolution codes here
    a = img_in
    gk = cv2.getGaussianKernel(21, 5)
    gk = gk * gk.T

    def ft(a, newsize=None):
        dft = np.fft.fft2(np.float64(a), newsize)
        return np.fft.fftshift(dft)

    def ift(shift):
        f_ishift = np.fft.ifftshift(shift)
        img_back = np.fft.ifft2(f_ishift)
        return np.abs(img_back)

    imf = ft(a, (a.shape[0], a.shape[1]))  # make sure sizes match
    gkf = ft(gk, (a.shape[0], a.shape[1]))  # so we can multiple easily
    imconvf = imf / gkf

    # now for example we can reconstruct the blurred image from its FT
    blurred = ift(imconvf)
    blurred = cv2.multiply(blurred, 255)
    img_out = blurred  # Deconvolution result

    return True, img_out


def Question2():

    # Read in input images a  = cv2.imread(img_in, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    input_image1 = cv2.imread(sys.argv[2], 0)
    input_image2 = cv2.imread(sys.argv[3],
                              cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Low and high pass filter
    succeed1, output_image1 = low_pass_filter(input_image1)
    succeed2, output_image2 = high_pass_filter(input_image1)

    # Deconvolution
    succeed3, output_image3 = deconvolution(input_image2)

    # Write out the result
    output_name1 = sys.argv[4] + "2.jpg"
    output_name2 = sys.argv[4] + "3.jpg"
    output_name3 = sys.argv[4] + "4.jpg"
    cv2.imwrite(output_name1, output_image1)
    cv2.imwrite(output_name2, output_image2)
    cv2.imwrite(output_name3, output_image3)

    return True


# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================


def laplacian_pyramid_blending(img_in1, img_in2):
    # Write laplacian pyramid blending codes here
    A = img_in1
    B = img_in2
    A = A[:, :A.shape[0]]
    B = B[:A.shape[0], :A.shape[0]]
    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpB.append(G)
    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in xrange(5, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in xrange(5, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)
    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
        LS.append(ls)
    # now reconstruct
    ls_ = LS[0]
    for i in xrange(1, 6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    # image with direct connecting each half
    real = np.hstack((A[:, :cols / 2], B[:, cols / 2:]))
    cv2.imwrite('Pyramid_blending2.jpg', ls_)
    cv2.imwrite('Direct_blending.jpg', real)
    img_out = ls_  # Blending result

    return True, img_out


def Question3():
    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR)

    # Laplacian pyramid blending
    succeed, output_image = laplacian_pyramid_blending(input_image1,
                                                       input_image2)

    # Write out the result
    output_name = sys.argv[4] + "5.jpg"
    cv2.imwrite(output_name, output_image)

    return True


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) < 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])

        if (question_number == 1 and not (len(sys.argv) == 4)):
            help_message()
            sys.exit()
        if (question_number == 2 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number == 3 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
            print("Input parameters out of bound ...")
            sys.exit()

    function_launch = {
        1: Question1,
        2: Question2,
        3: Question3,
    }

    # Call the function
    function_launch[question_number]()
