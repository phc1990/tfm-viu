"""Analysis input module."""


from enum import Enum


class Input(Enum):
    """Subset of available inputs."""
    DETECTED        = 1
    DUBIOUS         = 2
    NOT_DETECTED    = 3
    REPEAT          = 4
    CLOSE           = 5


class Interface:
    """Analysis input interface."""
    
    def message(self, message: str):
        """Sends a message to the input interface.

        Args:
            message (str): message to be displayed
        """
        pass
    
    def listen(self) -> Input:
        """Waits and listens for inputs.

        Returns:
            Input: received input
        """
        pass
    
    
class StdIoInterface(Interface):
    """Input interface based on standard input ('stdin') and 
    standard output ('stdout').
    """
    
    _DETECTED       = ["y", "yes", "Y", "YES"]
    _DUBIOUS        = ["d", "D"]
    _NOT_DETECTED   = ["n", "no", "N", "NO"]
    _REPEAT         = ["r", "repeat", "R", "REPEAT"]
    _CLOSE          = ["c", "close", "C", "CLOSE"]
    
    def message(self, message: str):
        """See base class."""
        print(message)
    
    def listen(self) -> Input:
        """See base class."""
        std_input = input()
        
        if std_input in self._DETECTED:
            return Input.DETECTED
        elif std_input in self._DUBIOUS:
            return Input.DUBIOUS
        elif std_input in self._NOT_DETECTED:
            return Input.NOT_DETECTED
        elif std_input in self._REPEAT:
            return Input.REPEAT
        elif std_input in self._CLOSE:
            return Input.CLOSE
        else:
            self.message('Invalid input. Available inputs:')
            self.message('Detection    : ' + ' '.join(self._DETECTED))
            self.message('Dubious      : ' + ' '.join(self._DUBIOUS))
            self.message('No detection : ' + ' '.join(self._NOT_DETECTED))
            self.message('Repeat       : ' + ' '.join(self._REPEAT))
            self.message('Close        : ' + ' '.join(self._CLOSE))
            return self.listen()
    