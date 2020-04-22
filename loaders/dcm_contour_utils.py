import json
import os
from collections import defaultdict

import dicom
import scipy
import numpy as np

class DicomImage(object):
    """
    Object for basic information of a DICOM image.
    """
    def __init__(self, dcm_image_file):
        print('Reading ' + dcm_image_file)
        assert os.path.exists(dcm_image_file)
        dcm_image = dicom.read_file(dcm_image_file)

        # this is unique for all images of the same patient
        self.patient  = str(dcm_image.PatientName) if hasattr(dcm_image, 'PatientName') else None

        # this is a series identifier
        self.series   = int(dcm_image.SeriesNumber) if hasattr(dcm_image, 'SeriesNumber') else None

        # the instance within the series
        self.instance = int(dcm_image.InstanceNumber) if hasattr(dcm_image, 'InstanceNumber') else None
        self.id       = None  # a unique id of the image
        self.resolution = [float(i) for i in dcm_image.PixelSpacing] + [float(dcm_image.SpacingBetweenSlices)]
        self.image    = dcm_image.pixel_array
        self.age      = dcm_image.PatientAge

    def save(self, folder):
        scipy.misc.imsave(folder + '/image_%s.png' % str(self.id), self.image)
        np.savez_compressed(folder + '/image_%s' % str(self.id), self.image)


class Coordinates(object):
    """
    Class for storing contours for a cardiac phase
    """
    def __init__(self):
        self.endo  = None
        self.epi   = None

class Contour(object):
    """
    Simple object for basic information of a contour
    """
    def __init__(self, contour_file):
        self.contour_file = contour_file
        self.patient_name       = None  # patient name. Also the parent folder of the images
        self.series             = None
        self.series_description = None  # the common identifier of the folders that contain the images
        self.coordinates        = defaultdict(lambda: defaultdict(lambda: Coordinates())) # a dictionary of Coordinates keyed by slice id and phase
        self.gender             = None
        self.birth_date         = None
        self.study_date         = None
        self.weight             = None
        self.height             = None
        self.age                = None
        self.es                 = None
        self.ed                 = None
        self.read_file()


    def read_file(self):
        with open(self.contour_file, 'r') as fd:
            last_slice = (-1, -1) # (slice, phase)
            while True:
                l = fd.readline()
                if l == '': break

                if 'Patient_name=' in l:
                    self.patient_name = l.split('Patient_name=')[1].split('\n')[0]

                if 'Series=' in l:
                    self.series = l.split('Series=')[1].split('\n')[0]

                if 'Series_description=' in l:
                    self.series_description = l.split('Series_description=')[1].split('/')[0]\
                        .strip().replace(' ', '_').replace('.', '_')

                if 'Patient_gender' in l:
                    self.gender = l.split('Patient_gender=')[1].split('\n')[0]

                if 'birth_date' in l:
                    self.birth_date = l.split('Birth_date=')[1].split('\n')[0]

                if 'Study_date' in l:
                    self.study_date = l.split('Study_date=')[1].split('\n')[0]

                if 'Patient_weight' in l:
                    self.weight = l.split('Patient_weight=')[1].split('\n')[0]

                if 'Patient_height' in l:
                    self.height = l.split('Patient_height=')[1].split('\n')[0]

                if 'manual_lv_es_phase' in l:
                    self.es = int(l.split('manual_lv_es_phase=')[1].split('\n')[0]) + 1 # images are 1-indexed

                if 'manual_lv_ed_phase' in l:
                    self.ed = int(l.split('manual_lv_ed_phase=')[1].split('\n')[0]) + 1 # images are 1-indexed

                # create a tuple of floats from a x-y coord pair.
                if '[XYCONTOUR]' in l:
                    header = fd.readline().split(' ')  # e.g. 1 0 0  1.0
                    slice = int(header[0])
                    phase = int(header[1])  # usually 0 -> ED, 7 -> ES
                    contour_type = int(header[2])  # 0 -> endo, 1 -> epi

                    # Initialise ES and ED if not explicitly defined in the contour file
                    if phase < 2 and self.ed is None:
                        self.ed = phase
                    if phase > 2 and self.es is None:
                        self.es = phase

                    num_coords = int(fd.readline())

                    parse_coord = lambda x: (float(x.split(' ')[0]), float(x.split(' ')[1]))
                    coords = [parse_coord(fd.readline()) for i in range(num_coords)]  # read coordinates

                    cc = self.coordinates[slice][phase]
                    if contour_type == 0:   # Coordinate for endo
                        cc.endo = coords
                    elif contour_type == 1: # Coordinate for epi
                        cc.epi = coords
                    self.coordinates[slice][phase] = cc

    def save(self, folder):
        with open(folder + '/contour.json', 'w') as outfile:
            d = self.__dict__.copy()
            d['coordinates'] = None
            json.dump(d, outfile)
