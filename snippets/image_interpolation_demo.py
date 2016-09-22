import numpy as np
import scipy.ndimage as ni
import pylab as pl
from scipy import stats

def add_noise(image, fft_S=24., fft_amp=64., poisson_lambda=128., gaussian_std= 16.):
    noise_image = np.random.randn(*image.shape)
    MAX = np.max(noise_image)
    fft_image = np.fft.fftshift(np.fft.fft2(noise_image))
    imp = np.zeros(image.shape)
    imp[int(image.shape[0] * 0.5), int(image.shape[1] * 0.5)] = 1.
    gaus = ni.filters.gaussian_filter(imp, fft_S)
    fft_image *= gaus
    noise_image = np.real(np.fft.ifft2(np.fft.fftshift(fft_image)))
    image += noise_image * MAX / np.max(noise_image) * fft_amp

    image += np.random.poisson(poisson_lambda, image.shape)

    image += np.abs(np.random.randn(*image.shape)*gaussian_std)
    return image


def generate_ring(image_shape, n_points=64, radius = 16, sigma=1., smoothness=1., intensity_factor=1.):
    cY,cX = 0.5*np.array(image_shape)
    ring = radius + np.random.randn(n_points)*radius/6.
    ring = ni.filters.gaussian_filter1d(ring,sigma=smoothness,mode='wrap')
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    ring_image = np.zeros(image_shape)
    for i,(angle,R) in enumerate(zip(angles,ring)):
        x0 = int(image_shape[1]*0.5 + R*np.cos(angle))
        y0 = int(image_shape[0]*0.5 + R*np.sin(angle))
        for x in range(-3,3+1):
            for y in range(-3,3+1):
                hx = stats.norm(0, sigma).pdf(x)
                hy = stats.norm(0, sigma).pdf(y)
                ring_image[y0+y,x0+x] += hx*hy*2500*intensity_factor

    # ni.filters.gaussian_filter(ring_image,sigma=sigma,output=ring_image)
    coordinates = [[cY+r*np.cos(a) for r,a in zip(ring,angles)],[cY+r*np.sin(a) for r,a in zip(ring,angles)],]
    return ring_image, coordinates


def map_image_to_polar_coords(image, n_rays, radius_range, n_radii):
    c0, c1 = 0.5 * np.array(np.shape(image))
    zoom_r = n_radii/(radius_range[1]-radius_range[0])
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=True)
    radii = np.linspace(0, radius * 2, int(radius * 2 * zoom_r))
    coordinatesX = np.array([[c0 + r_ * np.cos(angles), ] for r_ in radii])
    coordinatesY = np.array([[c1 + r_ * np.sin(angles), ] for r_ in radii])
    polar_image = ni.map_coordinates(image, coordinates=[coordinatesY, coordinatesX]).reshape(-1, n_rays)
    return polar_image


image_shape = (128,128)
radius = 16
ring, coordinates = generate_ring(image_shape, radius=radius)
pl.figure()
pl.subplot(221)
pl.imshow(ring,interpolation='nearest',cmap=pl.cm.viridis)
pl.hold(True)
xl = pl.xlim()
yl = pl.ylim()
pl.plot(coordinates[0],coordinates[1],':r')
pl.xlim(xl)
pl.ylim(yl)

pl.subplot(222)
ring = ni.filters.gaussian_filter(ring,sigma=1.5)
image = add_noise(ring)
pl.imshow(image,interpolation='nearest',cmap=pl.cm.viridis)
pl.hold(True)
xl = pl.xlim()
yl = pl.ylim()
pl.plot(coordinates[0],coordinates[1],':r')
pl.xlim(xl)
pl.ylim(yl)


pl.subplot(212)
n_rays = 128
radius_range = [0,2*radius]
n_radii = 3.*radius
zoom_r = n_radii/(radius_range[1]-radius_range[0])
polar_image = map_image_to_polar_coords(image, n_rays, radius_range, n_radii)
pl.imshow(polar_image,interpolation='nearest',cmap=pl.cm.viridis)
c0, c1 = 0.5 * np.array(np.shape(image))
xl = pl.xlim()
yl = pl.ylim()
coords_R = [np.sqrt((cy-c1)**2+(cx-c0)**2)*zoom_r-0.5 for cy,cx in zip(coordinates[0],coordinates[1])]
coords_angles = [np.arctan2(cy-c1,cx-c0) for cy,cx in zip(coordinates[0],coordinates[1])]
coords_angles = [((a-coords_angles[0]) % (2*np.pi))/(2*np.pi)*n_rays-1.5 for a in coords_angles][::-1]
pl.plot(coords_angles,coords_R,'.r')
pl.xlim(xl)
pl.ylim(yl)
pl.title("Polar Coordinates Representation")
pl.xlabel("Angles")
pl.ylabel("Radii")
pl.show()