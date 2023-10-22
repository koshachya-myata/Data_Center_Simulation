"""Helper classes for socket EnergyPlus work."""
import os
import socket
from  xml.etree import ElementTree


class SocketBuilder:
    """Helper class that creates socket.cfg files \
       for all EnergyPlus instances."""

    def __init__(self, path: str):
        """
        Args:
            path (str): path to main folder where we need to store the
                        socket.cfg file.
        """
        self.path = path

    def build(self):
        # work in the same directory as the input
        with ChangeDirClassWrapper(self.path):
            configs = []
            port = self.get_free_port()
            configs.append(port)
            # write the port configuration to socket.cfg
            self.build_XML(port)
        return configs

    def get_free_port(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
        s.close()
        return port

    def build_XML(self, port):
        # format of the XML is dictacted by EnergyPlus
        tree = ElementTree.ElementTree()
        bcvtb_client = ElementTree.Element("BCVTB-client")
        ipc = ElementTree.SubElement(bcvtb_client, "ipc")
        socket_ele = ElementTree.SubElement(ipc, "socket")
        socket_ele.set("port", str(port))
        socket_ele.set("hostname", "localhost")
        tree._setroot(bcvtb_client)
        tree.write("socket.cfg", encoding="ISO-8859-1")


class ChangeDirClassWrapper:
    """Context manager for changing the current working directory."""

    def __init__(self, new_path):
        self.newPath = os.path.expanduser(new_path)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
