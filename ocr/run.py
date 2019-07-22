import ocr

print(ocr.text_from_image_file('test-eng.PNG','eng')) # ดึงไฟล์ test-eng.PNG เข้ามา โดยกำหนดเป็นภาษาอังกฤษ
print(ocr.text_from_image_file('test-thai.PNG','tha')) # ดึงไฟล์ test-thai.PNG เข้ามา โดยกำหนดเป็นภาษาไทย
print(ocr.text_from_image_file('4.PNG','eng')) # กำหนดเป็นภาษาอังกฤษ
print(ocr.text_from_image_file('5.PNG','eng')) # กำหนดเป็นภาษาอังกฤษ
print(ocr.text_from_image_file('6.PNG','eng')) # กำหนดเป็นภาษาอังกฤษ
print(ocr.text_from_image_file('7.PNG','eng')) # กำหนดเป็นภาษาอังกฤษ
