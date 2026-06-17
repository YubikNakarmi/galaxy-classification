provider "azurerm" {
    
  features {}
  resource_provider_registrations = "none"

 
}


resource "azurerm_resource_group" "rg" {
    name     = var.resource_group_name
    location = var.location



}



resource "azurerm_storage_container" "example" {
  name                  = "content"
  storage_account_id    = azurerm_storage_account.example.id
  container_access_type = "private"
}

#----------storage-----------------
resource "azurerm_storage_account" "acc-name" {
  name                     = "${var.storage_account_name}"
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = "${var.location}"
  account_tier             = "Standard"
  account_replication_type = "LRS"
}


#-------network----------------
resource "azurerm_virtual_network" "vnet" {
    name     = "${var.resource_group_name}-vnet"
    address_space = ["10.0.0.0/16"] #vnet address allocation
    location = azurerm_resource_group.rg.location
    resource_group_name = azurerm_resource_group.rg.name
    
    }

resource "azurerm_subnet" "subnet" {
    name = "${var.resource_group_name}-subnet"
    resource_group_name = azurerm_resource_group.rg.name
    virtual_network_name = azurerm_virtual_network.vnet.name
    address_prefixes = ["10.0.1.0/24"]#subnet address allocation
    }

resource "azurerm_public_ip" "pip" {
    name = "${var.resource_group_name}-pip"
    location = azurerm_resource_group.rg.location
    resource_group_name = azurerm_resource_group.rg.name
    allocation_method = "Static" #Standard SKU requires static allocation
  
}

resource "azurerm_network_interface" "nic" {
    name = "${var.resource_group_name}-nic"
    location = azurerm_resource_group.rg.location
    resource_group_name = azurerm_resource_group.rg.name

    ip_configuration {
        name = "internal"
        subnet_id = azurerm_subnet.subnet.id
        private_ip_address_allocation = "Dynamic"#dynamic ip to reduce cost
    }
  
}

resource "azurerm_network_security_group" "nsg" {
    name = "${var.resource_group_name}-nsg"
    location = azurerm_resource_group.rg.location
    resource_group_name = azurerm_resource_group.rg.name
}

resource "azurerm_network_security_rule" "mlflow" {
    name = "Allow-MLflow"
    priority = 1003
    direction = "Inbound"
    access = "Allow"
    protocol = "Tcp"
    source_port_range = "*"
    destination_port_range = "5000"
    source_address_prefix = var.my-ip #restricting mlflow access to your public ip for security
    destination_address_prefix = "*"
    network_security_group_name = azurerm_network_security_group.nsg.name
    resource_group_name = azurerm_resource_group.rg.name
} 
