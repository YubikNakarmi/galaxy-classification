variable "resource_group_name" {
  description = "Resource grp name"
  type        = string
}


variable "mlflow_password"{
  description = "Password for mlflow"
  type        = string
  default     = "CHANGE_ME"
}
variable "storage_account_name" {
  description = "storage acc name"
  type = string
  
}

variable "vm_type"{
  description = "type of VM to serve"
  type = string
  default="Standard_B2s" #fallback to cheapest vm

}

variable "location" {
  description = "Location for the resource group"
  type        = string
  default     = "central india"
}

variable "vmname" {
  description = "Name of the virtual machine"
  type        = string
  default     = "train_vm"
}

variable "admin-name" {
    description = "User name for VM admin"
    type        = string
    default     = "azureuser"
  
}

variable "my-ip" {
    description = "Public IP address for NSG rule (CIDR format)"
    type        = string
    default     = "0.0.0.0/0"
}

variable "public-key" {
    description = "SSH public key"
    type        = string
    default     = "" #-var="ssh_public_key=$SSH_PUBLIC_KEY"
}