import base64

def save_jpg(file_path, data, process_selection):
    try:
        if process_selection == 0:
            with open(file_path, 'wb') as o:  
                o.write(base64.b64decode(data))
    except Exception as e:
        print(f"origin image save(jpg) error {e}")
        
    try:
        if process_selection == 1:
            with open(file_path + 'c_target.jpg', 'wb') as c:
                c.write(base64.b64decode(data))   
    except Exception as e:
        print(f"origin image save(jpg) error {e}") 
