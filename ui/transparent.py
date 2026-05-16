from PIL import Image

def make_transparent(file_path):
    try:
        img = Image.open(file_path).convert("RGBA")
        datas = img.getdata()
        
        newData = []
        for item in datas:
            # White-ish pixels become transparent
            if item[0] > 230 and item[1] > 230 and item[2] > 230:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
                
        img.putdata(newData)
        img.save(file_path, "PNG")
        print(f"Processed {file_path}")
    except Exception as e:
        print(f"Error on {file_path}: {e}")

make_transparent('knight.png')
make_transparent('rook.png')
