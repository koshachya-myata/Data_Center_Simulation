"""EnergyPlus contoller class."""

from typing import Union
import os
import socket
import subprocess

from . import pyEpError


class EplusController:
    """EnergyPlus contoller class."""

    def __init__(self, ip: str, port: Union[int, str], idf_file: str,
                 weather: str, eplus_path: Union[str, None] = None):
        """Initialize EnergyPlus process and socket."""
        self.init_eplus_process(idf_file, weather, eplus_path)
        self.init_socket(ip, port)

    def init_eplus_process(self, idf_file: str, weather: str,
                           eplus_path: Union[str, None] = None):
        """
        Initialize EnergyPlus process.

        Args:
            idf_file (str): path to .idf file.
            weather (str): path to weather file.
            eplus_path (Union[str, None], optional): path to EnergyPlus.
                                                     Defaults to None.
        """
        log_file = open("epluslog.txt", "w")
        set_bcvtb_home()
        if eplus_path is None:
            global eplus_dir
            if eplus_dir is None:
                raise pyEpError.MissingEpPathError
            if os.name == 'nt':
                eplus_path = os.path.join(eplus_dir, 'energyplus.exe')
            else:
                eplus_path = os.path.join(eplus_dir, 'energyplus')
        idf_dir = os.path.dirname(idf_file)
        os.chdir(idf_dir)

        if os.name == 'nt':  # for windows
            print('Windows support may not work')
            idf_path = os.path.join(os.path.dirname(__file__), idf_file)
            weather_path = os.path.join(os.path.dirname(__file__), weather)
            self.p = subprocess.Popen([eplus_path, '-w', weather_path,
                                       idf_path], stdout=log_file)

        else:  # for linux/mac
            idf_path = os.path.join(os.path.dirname(__file__), idf_file)
            weather_path = os.path.join(os.path.dirname(__file__), weather)
            self.p = subprocess.Popen([eplus_path, '-w', weather_path,
                                       idf_path], stdout=log_file)

        print('Using EnergyPlus executable: ' + eplus_path)
        print('Using IDF file: ' + idf_file)
        print('Creating EnergyPlus Process: ' + eplus_path + ' -w ' +
              weather + ' ' + idf_path)

    def init_socket(self, ip: str, port: Union[str, int]):
        """
        Initialize socker for EnergyPlus. Set connection to self.remote.

        Args:
            ip (str): ip.
            port (Union[str, int]): port.
        """
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((ip, port))
        print('Started waiting for connection on %s %s' % (ip, port))
        s.listen(1)
        remote, address = s.accept()
        self.remote = remote
        print('Got connection from Host ' + str(address[0]) +
              ' Port ' + str(address[1]))

    def close(self):
        """Close EnergyPluis process."""
        print('Closing EnergyPlus process')
        self.write('2 1\n')
        self.remote.shutdown(socket.SHUT_RDWR)
        self.remote.close()

    def read(self) -> str:
        """
        Read packet form EnergyPlus.

        Returns:
            str: packet data.
        """
        data = ''
        try:
            while True:
                packet = self.remote.recv(1024)
                packet = packet.decode('utf-8')
                data = data + packet
                if "\n" in packet:  # \n is end flag
                    break

        except socket.error:
            print('Socket Error')
            raise pyEpError.EpReadError

        return data

    def write(self, packet: str):
        """
        Write packet to EnegyPlus remote connection.

        Args:
            packet (str): packet data.
        """
        packet = packet.encode('utf-8')
        self.remote.send(packet)


def set_bcvtb_home():
    """Set BCVTB_HOME environ path."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bcvtb')
    os.environ['BCVTB_HOME'] = path  # visible in this process + all children
    print('Setting BCVTB_HOME to ', path)


def set_eplus_dir(path: str):
    """
    Set global eplus_dir to  path.

    Args:
        path (str): path.
    """
    global eplus_dir

    eplus_dir = path
