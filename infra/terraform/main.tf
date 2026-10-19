provider "azurerm" {
    
  features {}
  resource_provider_registrations = "none"

 
}


resource "azurerm_resource_group" "rg" {
    name     = var.resource_group_name
    location = var.location


}




#----------storage-----------------
resource "azurerm_storage_account" "acc-name" {
  name                     = "${var.storage_account_name}"
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = "${var.location}"
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

resource "azurerm_storage_container" "mlflow" {
  name                  = "mlflow"
  storage_account_id = azurerm_storage_account.acc-name.id
  container_access_type = "private"
}

resource "azurerm_storage_container" "dvc-storage" {
  name                  = "dvc-storage"
  storage_account_id = azurerm_storage_account.acc-name.id
  container_access_type = "private"
}

resource "azurerm_storage_container" "logs" {
  name                  = "galaxy-classification-logs"
  storage_account_id = azurerm_storage_account.acc-name.id
  container_access_type = "private"
}

resource "azurerm_storage_container" "experiments" { #for checkpoints
  name                  = "galaxy-experiments"
  storage_account_id = azurerm_storage_account.acc-name.id
  container_access_type = "private"
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

#---------VM--------------------------------------------------

resource "azurerm_linux_virtual_machine" "vm" {
    name = "${var.resource_group_name}-vm"
    resource_group_name = azurerm_resource_group.rg.name
    location = azurerm_resource_group.rg.location
    size = "${var.vm_type}" 
    admin_username = var.admin-name
    network_interface_ids = [azurerm_network_interface.nic.id]

    admin_ssh_key {
      username = var.admin-name
      public_key =  var.public-key #using existing ssh key for authentication
    }
    os_disk {
        caching = "ReadWrite"
        storage_account_type = "Standard_LRS"
    }
    source_image_reference {
        publisher = "Canonical"
        offer = "ubuntu-24_04-lts"
        sku = "server"
        version = "latest"
    }

    custom_data = filebase64("${path.module}/setup.sh") #cloud init script to install docker and start the container

    lifecycle {
      ignore_changes = [custom_data]
    }
}

