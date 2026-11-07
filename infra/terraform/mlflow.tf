provider "azurerm" {
    
  features {}
  resource_provider_registrations = "none"

 
}



resource "azurerm_public_ip" "pip-mlflow" {
    name = "mlflow-pip"
    location = azurerm_resource_group.rg.location
    resource_group_name = azurerm_resource_group.rg.name
    allocation_method = "Static" #Standard SKU requires static allocation
  
}

resource "azurerm_network_interface" "nic-mlflow"{
    name = "mlflow-nic"
    location = azurerm_resource_group.rg.location
    resource_group_name = azurerm_resource_group.rg.name

    ip_configuration {
        name = "internal"
        subnet_id = azurerm_subnet.subnet.id
        private_ip_address_allocation = "Dynamic"
    }
}

resource "azurerm_linux_virtual_machine" "mlflow-vm" {

    name = "mlflow-vm"
    resource_group_name = azurerm_resource_group.rg.name
    location = azurerm_resource_group.rg.location
    size = "B2als v2"
    admin_username = var.admin-name
    network_interface_ids = [azurerm_network_interface.nic.id]

    admin_ssh_key {
        username = var.admin-name
        public_key = file(var.public-key)
    }

    os_disk {
        caching              = "ReadWrite"
        storage_account_type = "Standard_LRS"
    }

    source_image_reference {
        publisher = "Canonical"
        offer = "ubuntu-24_04-lts"
        sku = "server"
        version = "latest"


    }

    custom_data = base64encode(
    replace(file("${path.module}/setup-mlflow.sh"), "CHANGE_ME", var.mlflow_password)
    )

}