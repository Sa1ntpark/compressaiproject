import io
from astropy.io import fits
import numpy as np
from PIL import Image
from astropy.io import fits

def byte_scale(img):
    img_min = img.min()
    img_max = img.max()
    scaled_img = (img - img_min) / (img_max - img_min) * 255.0
    return np.clip(scaled_img, 0, 255).astype(np.uint8)

def decompress_jpeg_to_array(buffer):
    image = Image.open(buffer)
    decompressed_array = np.array(image)

    return decompressed_array

def compress_image_as_jpeg(image, quality2):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality2)
    compressed_size = buffer.tell()
    buffer.seek(0)

    return buffer, compressed_size * 8

def compress_image_as_jpeg2000(image, irreversible=True):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG2000", irreversible=irreversible)
    compressed_size = buffer.tell()
    buffer.seek(0)

    return buffer, compressed_size * 8 

def compress_and_restore(data, method, quality2):
    quality2 = int(quality2)
    data = byte_scale(data)

    if method == 'jpeg':
        data = Image.fromarray(data).convert('RGB')
        buffer, compressed = compress_image_as_jpeg(data, quality2=quality2)
        restored = decompress_jpeg_to_array(buffer)
        restored = restored.transpose(2, 0, 1)[np.newaxis, ...]
#         data = np.array(data)
#         data_min, data_max = data.min(), data.max()
#         data_scaled = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
#         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
#         _, compressed = cv2.imencode('.jpg', data_scaled, encode_param)
#         restored = cv2.imdecode(compressed, cv2.IMREAD_UNCHANGED)
#         restored = (restored.astype(np.float32) / 255) * (data_max - data_min) + data_min
        
    elif method == 'jpeg2000':
        data = Image.fromarray(data).convert('RGB')
        buffer, compressed = compress_image_as_jpeg2000(data)
        restored = decompress_jpeg_to_array(buffer)
        restored = restored.transpose(2, 0, 1)[np.newaxis, ...]
#         data = np.array(data)
#         encode_param = [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), quality * 1000 // 100]
#         _, compressed = cv2.imencode('.jp2', data, encode_param)
#         restored = cv2.imdecode(compressed, cv2.IMREAD_UNCHANGED)
        
    elif method == 'rice':
        compressed_hdu = fits.CompImageHDU(data=data, compression_type='RICE_1', tile_shape=(512, 512), quantize_level=opt.quantization)
        with io.BytesIO() as buffer:
            compressed_hdu.writeto(buffer)
            compressed_size = buffer.tell()
            buffer.seek(0)
            
        compressed = compressed_size * 8
        restored = compressed_hdu.data
        restored = restored[np.newaxis, np.newaxis, ...]

    return compressed, restored