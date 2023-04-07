"""Configuration utility module.
"""

import sys
import logging
import configparser


class Config:
    """Configuration wrapper.
    """

    def __init__(self, data) -> None:
        """Constructor.

        Args:
            data (_type_): configuration data, to be accessed as a map[key].
        """
        self.data = data


    def child(self, child_name: str):
        """Returns a child configuration object.

        Args:
            child_name (str): child name.

        Returns:
            _type_: the child configuration.
        """
        return Config(self.get_field(child_name))
    
    def raise_unrecognised_field_value(self, field_name: str, field_value: str):
        """Raises an exception for an unrecognised field's value.

        Args:
            field_name (str): field name.
            field_value (str): unrecognised field value.

        Raises:
            Exception: an Exception instance.
        """
        raise Exception('Unrecognised value "' + field_value + '" for field "' + field_name + '"')

    def get_field(self, field_name: str):
        """Returns the field's value, raising an exception if not found

        Args:
            field_name (str): field's value.

        Raises:
            Exception: if the field is missing.
            Exception: if the field is empty.

        Returns:
            _type_: the value of the field.
        """
        if field_name not in self.data:
            raise Exception('Missing field "' + field_name + '"')
        elif self.data[field_name] == '':
            raise Exception('Empty field "' + field_name + '"')
        
        return self.data[field_name]

    def get_boolean(self, field_name: str) -> bool:
        """Returns the field's value as a Boolean, raising an exception when not able to.

        Args:
            field_name (str): field's name.

        Returns:
            bool: the boolean value of the field.
        """
        value = self.get_field(field_name)
        if value == 'TRUE':
            return True
        elif value == 'FALSE':
            return False
        
        self.raise_unrecognised_field_value(field_name, value)
        
    def get_float(self, field_name: str) -> float:
        """Returns the field's value as a float, raising an exception when not able to.

        Args:
            field_name (str): field's name.

        Returns:
            float: the float value of the field.
        """
        value = self.get_field(field_name)
        return float(value)
    
    def get_int(self, field_name: str) -> int:
        """Returns the field's value as a int, raising an exception when not able to.

        Args:
            field_name (str): field's name.

        Returns:
            int: the int value of the field.
        """
        value = self.get_field(field_name)
        return int(value)
    

def config_from_sys_ars() -> Config:
    """Creates a Config instance from System arguments. It assumes that there
    are only 2 arguments:
    
    - sys.argv[0]: name of the script being executed
    - sys.argv[1]: filepath of the configuration file
    
    It will also configure the logging. It assumes that the following fields 
    are specified in the config root level:
    
    - 'LOG_LEVEL': with either 'DEBUG' or 'INFO'

    Raises:
        Exception: if there are too few arguments
        Exception: if there are too many arguments

    Returns:
        Config: a new config instance.
    """
    args = sys.argv 

    if len(args) < 2:
        raise Exception('Missing .ini configuration file location.')
    elif len(args) > 2:
        raise Exception('Too many arguments.')
    
    data = configparser.ConfigParser(inline_comment_prefixes='#')
    data.read(args[1])
    
    config = Config(data=data)
    required = config.child('REQUIRED')
    
    log_level_config = required.get_field('LOG_LEVEL')
    log_level = None
    if log_level_config == 'DEBUG':
        log_level = logging.DEBUG
    elif log_level_config == 'INFO':
        log_level = logging.INFO
    else:
        config.raise_unrecognised_field_value('LOG_LEVEL', log_level_config)
        
    logging.basicConfig(level=log_level, format='%(asctime)s  %(levelname)s %(message)s')
    return config