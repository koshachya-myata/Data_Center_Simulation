import os
import socket
import subprocess
import sys

from eplus import pyEpError


class Eplus_controller:

    def __init__(self, ip, port, idf_file, weather, eplus_path=None):
        self.init_Eplus_process(idf_file, weather, eplus_path)
        self.init_socket(ip, port)
    
    def init_Eplus_process(self, idf_file, weather, eplus_path=None):
        log_file = open("epluslog.txt", "w")
        set_bcvtb_home()
        if eplus_path is None:
            global eplus_dir
            if eplus_dir is None:
                raise pyEpError.MissingEpPathError
            eplus_path = eplus_dir
        idf_dir = os.path.dirname(idf_file)
        os.chdir(idf_dir)

        if os.name == "nt":  # for windows
            print('FIXME')
            exit(1)

        else:  # for linux/mac
            eplus_script = eplus_path + "energyplus"
            idf_path = os.path.join(os.path.dirname(__file__), idf_file)
            weather_path = os.path.join(os.path.dirname(__file__), weather)
            self.p = subprocess.Popen([eplus_script, "-w", weather_path, idf_path], stdout=log_file)

        print("Using EnergyPlus executable: " + eplus_script)
        print("Using IDF file: " + idf_file)
        print("Creating EnergyPlus Process: " + eplus_script + " -w " + weather + " " + idf_path)
    
    def init_socket(self, ip, port):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((ip, port))
        print("Started waiting for connection on %s %s" % (ip, port))
        s.listen(1)
        remote, address = s.accept()
        self.remote = remote
        print("Got connection from Host " + str(address[0]) + " Port " + str(address[1]))

    def close(self):
        print("Closing EnergyPlus process")
        self.write("2 1\n")
        self.remote.shutdown(socket.SHUT_RDWR)
        self.remote.close()

    def read(self):
        data = ""
        try:
            while True:
                packet = self.remote.recv(1024)
                packet = packet.decode("utf-8")
                data = data + packet
                if "\n" in packet:  # \n is end flag
                    break

        except socket.error:
            print("Socket Error")
            raise pyEpError.EpReadError

        return data

    def write(self, packet):
        packet = packet.encode("utf-8")
        self.remote.send(packet)


def set_bcvtb_home():
    path = os.path.dirname(os.path.abspath(__file__)) + "/bcvtb"
    os.environ["BCVTB_HOME"] = path  # visible in this process + all children
    print("Setting BCVTB_HOME to ", path)


def set_eplus_dir(path):
    global eplus_dir

    if path is not None:
        if not path.endswith("/"):
            path = path + "/"

    eplus_dir = path


"""
Energy Plus Protocol Version 1 & 2:
Packet has the form:
      v f dr di db t r1 r2 ... i1 i2 ... b1 b2 ...
where
  v    - version number (1,2)
  f    - flag (0: communicate, 1: finish, -10: initialization error,
               -20: time integration error, -1: unknown error)
  dr   - number of real values
  di   - number of integer values
  db   - number of boolean values
  t    - current simulation time in seconds (format %20.15e)
  r1 r2 ... are real values (format %20.15e)
  i1 i2 ... are integer values (format %d)
  b1 b2 ... are boolean values (format %d)
Note that if f is non-zero, other values after it will not be processed.
"""
