import numpy as np
from PIL import Image ,ImageChops , ImageOps
from PIL import ImageEnhance
import os

def make_thumb(image, size=(512, 512), pad=False):
	# http://stackoverflow.com/questions/9103257/resize-image-
	# maintaining-aspect-ratio-and-making-portrait-and-landscape-images-e
	image.thumbnail(size, Image.BILINEAR)
	image_size = image.size

	if pad:
		thumb = image.crop((0, 0, size[0], size[1]))

		offset_x = max((size[0] - image_size[0]) / 2, 0)
		offset_y = max((size[1] - image_size[1]) / 2, 0)

		thumb = ImageChops.offset(thumb, offset_x, offset_y)
	else:
		thumb = ImageOps.fit(image, size, Image.BILINEAR, (0.5, 0.5))

	return thumb

def load_image_and_process(img, output_shape=(512, 512),
						   prefix_path='',
						   transfo_params=None,
						   rand_values=None):
	im = Image.open(prefix_path + img , mode='r')

	sort_dim = list(np.sort(im.size))

	#dim_dst[0] = sort_dim[1] / 700.0
	#dim_dst[1] = sort_dim[0] / 700.0

	im_new = im

	# Dict to keep track of random values.
	chosen_values = {}

	if transfo_params.get('extra_width_crop', False):#jian qu hei se bu fen
		w, h = im_new.size

		
		cols_thres = np.where(
			np.max(np.max(np.asarray(im_new),axis=2),axis=0) > 35)[0]

		# Extra cond compared to orig crop.
		if len(cols_thres) > output_shape[0] // 2:
			min_x, max_x = cols_thres[0], cols_thres[-1]
		else:
			min_x, max_x = 0, -1

		im_new = im_new.crop((min_x, 0,
							  max_x, h))

	if transfo_params.get('crop_height', False):#Flase
		w, h = im_new.size

		if w > 1 and 0.98 <= h / float(w) <= 1.02:
			# "Normal" without height crop, do height crop.
			im_new = im_new.crop((0, int(0.05 * h),
								  w, int(0.95 * h)))

	if transfo_params.get('crop', False) and not \
			transfo_params.get('crop_after_rotation', False):#Flase
		if rand_values:
			do_crop = rand_values['do_crop']
		else:
			do_crop = transfo_params['crop_prob'] > np.random.rand()
		chosen_values['do_crop'] = do_crop

		if do_crop:
			out_w, out_h = im_new.size
			w_dev = int(transfo_params['crop_w'] * out_w)
			h_dev = int(transfo_params['crop_h'] * out_h)

			# If values are supplied.
			if rand_values:
				w0, w1 = rand_values['w0'], rand_values['w1']
				h0, h1 = rand_values['h0'], rand_values['h1']
			else:
				w0 = np.random.randint(0, w_dev + 1)
				w1 = np.random.randint(0, w_dev + 1)
				h0 = np.random.randint(0, h_dev + 1)
				h1 = np.random.randint(0, h_dev + 1)

			# Add params to dict.
			chosen_values['w0'] = w0
			chosen_values['w1'] = w1
			chosen_values['h0'] = h0
			chosen_values['h1'] = h1

			im_new = im_new.crop((0 + w0, 0 + h0,
								  out_w - w1, out_h - h1))

	# if transfo_params.get('new_gen', False):
	#     im_new = im_new.crop(im_new.getbbox())
	# im_new = im_new.resize(map(lambda x: x*2, output_shape),
	# resample=Image.BICUBIC)

	if transfo_params.get('shear', False):#Flase
		# http://stackoverflow.com/questions/14177744/how-does-
		# perspective-transformation-work-in-pil
		if transfo_params['shear_prob'] > np.random.rand():
			# print 'shear'
			# TODO: No chosen values because shear not really used.
			shear_min, shear_max = transfo_params['shear_range']
			m = shear_min + np.random.rand() * (shear_max - shear_min)
			out_w, out_h = im_new.size
			xshift = abs(m) * out_w
			new_width = out_w + int(round(xshift))
			im_new = im_new.transform((new_width, out_h), Image.AFFINE,
									  (1, m, -xshift if m > 0 else 0, 0, 1, 0),
									  Image.BICUBIC)

	if transfo_params.get('rotation_before_resize', False):#Flase
		if rand_values:
			rotation_param = rand_values['rotation_param']
		else:
			rotation_param = np.random.randint(
				transfo_params['rotation_range'][0],
				transfo_params['rotation_range'][1])
		chosen_values['rotation_param'] = rotation_param

		im_new = im_new.rotate(rotation_param, resample=Image.BILINEAR,
							   expand=transfo_params.get('rotation_expand',
														 False))
		if transfo_params.get('rotation_expand',
							  False):
			im_new = im_new.crop(im_new.getbbox())

	if transfo_params.get('crop_after_rotation', False):#xuan zhuan hou crop
		if rand_values:
			do_crop = rand_values['do_crop']
		else:
			do_crop = transfo_params['crop_prob'] > np.random.rand()
		chosen_values['do_crop'] = do_crop

		if do_crop:
			out_w, out_h = im_new.size
			w_dev = int(transfo_params['crop_w'] * out_w)
			h_dev = int(transfo_params['crop_h'] * out_h)

			# If values are supplied.
			if rand_values:
				w0, w1 = rand_values['w0'], rand_values['w1']
				h0, h1 = rand_values['h0'], rand_values['h1']
			else:
				w0 = np.random.randint(0, w_dev + 1)
				w1 = np.random.randint(0, w_dev + 1)
				h0 = np.random.randint(0, h_dev + 1)
				h1 = np.random.randint(0, h_dev + 1)

			# Add params to dict.
			chosen_values['w0'] = w0
			chosen_values['w1'] = w1
			chosen_values['h0'] = h0
			chosen_values['h1'] = h1

			im_new = im_new.crop((0 + w0, 0 + h0,
								  out_w - w1, out_h - h1))

	# im_new = im_new.thumbnail(output_shape, resample=Image.BILINEAR)
	if transfo_params.get('keep_aspect_ratio', False):####resize
		im_new = make_thumb(im_new, size=output_shape,
						   pad=transfo_params['resize_pad'])
	else:
		im_new = im_new.resize(output_shape, resample=Image.BILINEAR)
	# im_new = im_new.resize(output_shape, resample=Image.BICUBIC)
	# im_new = im_new.resize(map(lambda x: int(x * 1.2), output_shape),
	# resample=Image.BICUBIC)
	# im_new = im_new.crop(im_new.getbbox())

	if transfo_params.get('rotation', False) \
			and not transfo_params.get('rotation_before_resize', False):##xuan zhuan
		if rand_values:
			rotation_param = rand_values['rotation_param']
		else:
			rotation_param = np.random.randint(
				transfo_params['rotation_range'][0],
				transfo_params['rotation_range'][1])
		chosen_values['rotation_param'] = rotation_param

		im_new = im_new.rotate(rotation_param, resample=Image.BILINEAR,
							   expand=transfo_params.get('rotation_expand',
														 False))
		if transfo_params.get('rotation_expand',
							  False):
			im_new = im_new.crop(im_new.getbbox())

	# im_new = im_new.resize(output_shape, resample=Image.BICUBIC)
	if transfo_params.get('contrast', False):##dui bi du
		contrast_min, contrast_max = transfo_params['contrast_range']
		if rand_values:
			contrast_param = rand_values['contrast_param']
		else:
			contrast_param = np.random.uniform(contrast_min, contrast_max)
		chosen_values['contrast_param'] = contrast_param

		im_new = ImageEnhance.Contrast(im_new).enhance(contrast_param)

	if transfo_params.get('brightness', False):
		brightness_min, brightness_max = transfo_params['brightness_range']
		if rand_values:
			brightness_param = rand_values['brightness_param']
		else:
			brightness_param = np.random.uniform(brightness_min,
												 brightness_max)
		chosen_values['brightness_param'] = brightness_param

		im_new = ImageEnhance.Brightness(im_new).enhance(brightness_param)

	if transfo_params.get('color', False):
		color_min, color_max = transfo_params['color_range']
		if rand_values:
			color_param = rand_values['color_param']
		else:
			color_param = np.random.uniform(color_min, color_max)
		chosen_values['color_param'] = color_param

		im_new = ImageEnhance.Color(im_new).enhance(color_param)

	if transfo_params.get('flip', False):
		if rand_values:
			do_flip = rand_values['do_flip']
		else:
			do_flip = transfo_params['flip_prob'] > np.random.rand()

		chosen_values['do_flip'] = do_flip

		if do_flip:
			im_new = im_new.transpose(Image.FLIP_LEFT_RIGHT)

	if output_shape[0] < 200 and False:
		# Otherwise too slow.
		# TODO: Disabled for now
		if 'rotation' in transfo_params and transfo_params['rotation']:
			if rand_values:
				rotation_param = rand_values['rotation_param2']
			else:
				rotation_param = np.random.randint(
					transfo_params['rotation_range'][0],
					transfo_params['rotation_range'][1])

			im_new = im_new.rotate(rotation_param, resample=Image.BILINEAR,
								   expand=False)
			# im_new = im_new.crop(im_new.getbbox())
			chosen_values['rotation_param2'] = rotation_param

	if transfo_params.get('zoom', False):
		if rand_values:
			do_zoom = rand_values['do_zoom']
		else:
			do_zoom = transfo_params['zoom_prob'] > np.random.rand()
		chosen_values['do_zoom'] = do_zoom

		if do_zoom:
			zoom_min, zoom_max = transfo_params['zoom_range']
			out_w, out_h = im_new.size
			if rand_values:
				w_dev = rand_values['w_dev']
			else:
				w_dev = int(np.random.uniform(zoom_min, zoom_max) / 2 * out_w)
			chosen_values['w_dev'] = w_dev

			im_new = im_new.crop((0 + w_dev,
								  0 + w_dev,
								  out_w - w_dev,
								  out_h - w_dev))

	# im_new = im_new.resize(output_shape, resample=Image.BILINEAR)
	if im_new.size != output_shape:
		im_new = im_new.resize(output_shape, resample=Image.BILINEAR)
	#im_new.save('/home/wanghao/trans/%s'%img)
	im_new = np.asarray(im_new).astype('float32') / 255
	#im_dst[:] = np.rollaxis(im_new.astype('float32'), 2, 0)

	im.close()
	del im

	return im_new


