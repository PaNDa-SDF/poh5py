#!/usr/bin/env python

"""Module to handle poh5 format file.

This module defines a class for poh5 format file.
"""

from __future__ import print_function, division
import sys
import h5py
import numpy as np

_POH5_VERSION = 94


class File(h5py.File):
    """ PoH5 file class, subclass of h5py.File."""

    def __init__(self, name, new=True, topo=0):
        if (new):
            h5py.File.__init__(self, name, mode='w')
            self.attrs.create('poh5_version', _POH5_VERSION, dtype='>i')
            self.attrs.create('num_of_var', 0, dtype='>i')
            self.attrs.create('grid_topology', topo, dtype='>i')
            self.gl = 0
            self.rl = 0
            self.nrgn = 0
        else:
            h5py.File.__init__(self, name, mode='r+')
            self.gl = self.attrs['glevel']
            self.rl = self.attrs['rlevel']
            self.nrgn = self.attrs['num_of_rgn']
            self.gall1d = 2**(self.gl-self.rl)+2
            self.gall = self.gall1d**2

        self.require_group('/Var')
        self.require_group('/Grd')
        self.flush()
        return None

    def set_global_attrs(self, gl, rl, npe=0, mpe=0, desc='', note=''):
        """ Set global attributes.

        Given number of total pe `npe` and my pe `mpe`,
        num_of_rgn and rngid[] are calc'ed and written to a file.

        If npe is 0, set is_complete as TRUE.

        Do NOT call me for existing file.
        """
        self.gl = gl
        self.rl = rl
        self.tot_rgn = 10*4**self.rl
        self.tot_rgnid = np.arange(self.tot_rgn)
        self.gall1d = 2**(self.gl-self.rl)+2
        self.gall = self.gall1d**2
        self.attrs.create('glevel', gl, dtype='>i')
        self.attrs.create('rlevel', rl, dtype='>i')

        if (npe == 0):
            is_complete = 1
            self.npe = 1
            self.mpe = 0
        else:
            is_complete = 0
            self.npe = npe
            self.mpe = mpe

        self.nrgn = self.tot_rgn/self.npe
        rgn_str = self.nrgn*self.mpe
        rgn_end = self.nrgn*self.mpe+self.nrgn
        mrgn = np.arange(rgn_str, rgn_end)

        self.attrs.create('is_complete', is_complete, dtype='>i')
        self.attrs.create('num_of_rgn', self.nrgn, dtype='>i')
        self.attrs.create('rgnid', mrgn, dtype='>i')
        self.attrs.create('description', desc, dtype='S64')
        self.attrs.create('note', note, dtype='S256')
        self.attrs.create('num_of_pe', npe, dtype='>i')
        self.attrs.create('my_pe', mpe, dtype='>i')
        self.flush()
        return None

    def show_global_attrs(self):
        """Just show global attributes, do nothing else."""
        sepr = "{0:*^50}" 
        form = "{0:>15}: {1}"
        print(sepr.format(" Global Attributes "))
        print(form.format("filename", self.filename))
        print(form.format("poh5_version", self.attrs['poh5_version']))
        print(form.format("description", self.attrs['description']))
        print(form.format("note", self.attrs['note']))
        print(form.format("glevel", self.attrs['glevel']))
        print(form.format("rlevel", self.attrs['rlevel']))
        print(form.format("is_complete", self.attrs['is_complete']))
        if (self.attrs['poh5_version'] > 93):
            print(form.format("num_of_pe", self.attrs['num_of_pe']))
            print(form.format("my_pe", self.attrs['my_pe']))
        print(form.format("num_of_rgn", self.attrs['num_of_rgn']))
        print(form.format("rgnid[]",""))
        print(self.attrs['rgnid'])
        print(sepr.format(""))
        return None

    def create_variable(self,
                        vname, nlayer, lname,
                        units=None, desc=None, note=None, dtype='>f8'):
        """Create new variable with attributes."""
        g = self.require_group('/Var/'+vname)
        g.attrs.create('varname', vname, dtype='S16')
        g.attrs.create('layername', lname, dtype='S16')
        g.attrs.create('num_of_layer', nlayer, dtype='>i')
        g.attrs.create('num_of_steps', 0, dtype='>i')
        g.attrs.create('units', units, dtype='S16')
        g.attrs.create('description', desc, dtype='S64')
        g.attrs.create('note', note, dtype='S64')

        # self.nrgn=self.attrs['num_of_rgn']
        d = g.require_dataset(
            'data',
            shape=(0, self.nrgn, nlayer, self.gall1d, self.gall1d),
            chunks=(1, 1, 1, self.gall1d, self.gall1d),
            compression='gzip', shuffle=True,
            maxshape=(None, self.nrgn, nlayer, self.gall1d, self.gall1d),
            dtype=dtype
        )

        dt = np.dtype([('start', ">i8"), ('end', '>i8')])
        t = g.require_dataset("time",
                              shape=(0,),
                              maxshape=(None,),
                              dtype=dt)
        self.flush()
        return None

    def add_variable(self, vname, ts, te, data):
        """Add variable data.

        Extend /Var/vname/{data,time} and add data,ts,te.
        """
        print("Adding {0} at time: {1!s}, {2!s}."
              .format(vname, ts, te))
        g = self.require_group('/Var/'+vname)
        d = g.get('data')
        t = g.get('time')

        new_t_size = d.shape[0]+1
        dslice = d.shape[1:]

        if (data.shape != dslice):
            form = "{0:>20}: {1}"
            print("poh5py.add_variable(): Shape mismatch:")
            print(form.format("data in file", dslice))
            print(form.format("data given", data.shape))
            sys.exit(1)

        d.resize(new_t_size, axis=0)
        d[new_t_size-1, ...] = data
        t.resize(new_t_size, axis=0)
        t[new_t_size-1] = (ts, te)
        self.flush()
        return None

    def create_hgrid(self):
        """Create new dataset for hgrid.

        Values are set by set_hgrid() or set_hgrid_from_file().
        """
        g = self.require_group('/Grd/Hgrid')
        d = g.require_dataset(
            'grd_x',
            shape=(3, self.nrgn, self.gall1d, self.gall1d),
            chunks=(1, 1, self.gall1d, self.gall1d),
            compression='gzip',
            shuffle=True,
            dtype='>f8')
        d = g.require_dataset(
            'grd_xt',
            shape=(3, 2, self.nrgn, self.gall1d, self.gall1d),
            chunks=(1, 1, 1, self.gall1d, self.gall1d),
            compression='gzip',
            shuffle=True,
            dtype='>f8')

        if (self.attrs['poh5_version'] < _POH5_VERSION):
            self.attrs['poh5_version'] = _POH5_VERSION
        self.flush()
        return None

    def set_hgrid(self, grd_x, grd_xt=None):
        """Set hgrid data.

        Existing data (if any) will be overwritten.
        shape(grd_x) must be (3,nrgn,gall1d,gall1d).
        shape(grd_xt) must be (3,2,nrgn,gall1d,gall1d).
        """
        g = self.require_group('/Grd/Hgrid')
        if (grd_x is not None):
            d = g.get('grd_x')
            d[:, :, :, :] = grd_x[:, :, :, :]

        if (grd_xt is not None):
            d = g.get('grd_xt')
            d[:, :, :, :, :] = grd_xt[:, :, :, :, :]
        self.flush()
        return None

    def set_hgrid_from_file(self, basename):
        """Set hgrid data from legacy format files."""

        (grd_x, grd_xt) = read_legacy_hgrid(
            basename, self.attrs['rgnid'], self.gall1d)
        self.set_hgrid(grd_x, grd_xt)
        self.flush()
        return None


################################
#       Oher utilities         #
################################

def cart2sphe(x, y, z, deg=False, calc_r=False):
    """ Convert from (x,y,z) to (lon, lat, [r]).

    Since numpy.arctan2 returns [-pi,pi] in radian,
    convert radian to degree if `deg` is True."""

    lon = np.arctan2(y, x)
    lat = np.arctan2(z, np.sqrt(x**2+y**2))

    if (deg):
        lon = np.degrees(lon)
        lat = np.degrees(lat)

    if (calc_r):
        r = np.sqrt(x**2 + y**2 + z**2)
        return(lon, lat, r)
    else:
        return(lon, lat)


def get_lib_version():
    return _POH5_VERSION


def show_license():
    license = """
Copyright(c) 2016 RIST.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of the "RIST" nor the names of its contributors may
  be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
    print(license)
    sys.exit(0)

################################
#      Internal function.      #
################################

def read_legacy_hgrid(basename, rgnids, gall1d, da=False):
    """Open and read legacy hgrid file.

    Returns (grd_x, grd_xt) data.
    grd_xt will be None if reading grd_xt fails.
    """

    if (not basename):
        return(None, None)

    head = ("head", "<i")
    tail = ("tail", "<i")

    grd_x = np.empty([3, len(rgnids), gall1d, gall1d], dtype='f8')
    grd_xt = np.empty([3, 2, len(rgnids), gall1d, gall1d], dtype='f8')

    for i, rgn in enumerate(rgnids):
        file = ''.join([basename, '.rgn', str(rgn).zfill(5)])
        try:
            fd = open(file)
        except IOError:
            mess = "No hgrid file: {0}, Do nothing."
            print(mess.format(file))
            return (None, None)

        ### Note that mkgcgrid calls GRD_output_hgrid() with
        ### bgrid_dump=.true. and da_access=.false.
        if (da):
            print("Implement This")
            sys.exit(1)
        else:
            dd = np.fromfile(
                fd, dtype=np.dtype([head, ("gall1d", ">i"), tail]), count=1)
            gall1d = dd[0]["gall1d"]

            ### read grd_x[3,lall,gall1d,gall1d]
            ff = str(gall1d**2)+'>f8'
            dt = np.dtype([head, ("body", ff), tail])
            dd = np.fromfile(fd, dtype=dt, count=3)
            grd_x[0, i, :, :] = dd[0]['body'].reshape(gall1d, gall1d)
            grd_x[1, i, :, :] = dd[1]['body'].reshape(gall1d, gall1d)
            grd_x[2, i, :, :] = dd[2]['body'].reshape(gall1d, gall1d)

            ### read grd_xt[3,2,lall,gall1d,gall1d]
            ff = str(2*gall1d**2)+'>f8'
            dt = np.dtype([head, ("body", ff), tail])
            try:
                dd = np.fromfile(fd, dtype=dt, count=3)
                grd_xt[0, :, i, :, :] = dd[0]['body'].reshape(2, gall1d, gall1d)
                grd_xt[1, :, i, :, :] = dd[1]['body'].reshape(2, gall1d, gall1d)
                grd_xt[2, :, i, :, :] = dd[2]['body'].reshape(2, gall1d, gall1d)
            except IOError:
                grd_xt = None

    return (grd_x, grd_xt)
